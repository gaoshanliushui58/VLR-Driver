import os
import json
import random
import datetime
import pathlib
import time
import imp
from collections import deque
import math
from copy import deepcopy

import yaml
import cv2
import torch
import carla
import numpy as np
from PIL import Image
from easydict import EasyDict
from torchvision import transforms

from leaderboard.autoagents import autonomous_agent
from team_code.planner import RoutePlanner, InstructionPlanner
from team_code.pid_controller import PIDController
from timm.models import create_model
from timm.data.carla_dataset import lidar_to_raw_features
import sys
from lavis.common.registry import registry

try:
    import pygame
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")


SAVE_PATH = os.environ.get("SAVE_PATH", 'eval')
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IS_BENCH2DRIVE = os.environ.get('IS_BENCH2DRIVE', None)

class DisplayInterface(object):
    def __init__(self):
        self._width = 1200
        self._height = 900
        self._surface = None

        pygame.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        self._display = pygame.display.set_mode(
            (self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("LMB2D Agent")

    def run_interface(self, input_data):
        rgb_front       = input_data['rgb_front']
        rgb_front_left  = input_data['rgb_front_left']
        rgb_front_right = input_data['rgb_front_right']
        rgb_back_left   = input_data['rgb_back_left']
        rgb_back_right  = input_data['rgb_back_right']
        rgb_center      = input_data['rgb_center']
        surface = np.zeros((900, 1200, 3),np.uint8)
        surface[:, :1200]           = rgb_front
        surface[:210, :280]         = rgb_front_left
        surface[:210, 920:1200]     = rgb_front_right
        surface[:210, 495:705]      = rgb_center
        surface[300:510, :280]      = rgb_back_left
        surface[300:510, 920:1200]  = rgb_back_right
        surface = cv2.putText(surface, input_data['time'], (20,710), cv2.FONT_HERSHEY_TRIPLEX,0.75,(0,0,255), 1)
        surface = cv2.putText(surface, input_data['meta_control'], (20,740), cv2.FONT_HERSHEY_TRIPLEX,0.75,(0,0,255), 1)
        surface = cv2.putText(surface, input_data['waypoints'], (20,770), cv2.FONT_HERSHEY_TRIPLEX,0.75,(0,0,255), 1)
        surface = cv2.putText(surface, input_data['instruction'], (20,800), cv2.FONT_HERSHEY_TRIPLEX,0.75,(0,0,255), 1)
        surface = cv2.putText(surface, input_data['notice'], (20,830), cv2.FONT_HERSHEY_TRIPLEX,0.75,(0,0,255), 1)

        surface = cv2.putText(surface, 'Front Left View', (60,245), cv2.FONT_HERSHEY_TRIPLEX,0.75,(139,69,19), 2)
        surface = cv2.putText(surface, 'Back Left View', (60,545), cv2.FONT_HERSHEY_TRIPLEX,0.75,(139,69,19), 2)
        surface = cv2.putText(surface, 'Focus View', (535,245), cv2.FONT_HERSHEY_TRIPLEX,0.75,(139,69,19), 2)
        surface = cv2.putText(surface, 'Front Right View', (980,245), cv2.FONT_HERSHEY_TRIPLEX,0.75,(139,69,19), 2)
        surface = cv2.putText(surface, 'Back Right View', (980,545), cv2.FONT_HERSHEY_TRIPLEX,0.75,(139,69,19), 2)
        

        surface[:210, 278:282] = [139,69,19]
        surface[:210, 493:497] = [139,69,19]
        surface[:210, 703:707] = [139,69,19]
        surface[:210, 918:922] = [139,69,19]
        surface[208:212, :280] = [139,69,19]
        surface[208:212, 920:1200] = [139,69,19]
        surface[208:212, 495:705] = [139,69,19]

        self._surface = pygame.surfarray.make_surface(surface.swapaxes(0,1))

        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))

        pygame.display.flip()
        pygame.event.get()
        return surface

    def _quit(self):
        pygame.quit()


def get_entry_point():
    return "LMB2DAgent"


class Resize2FixedSize:
    def __init__(self, size):
        self.size = size

    def __call__(self, pil_img):
        pil_img = pil_img.resize(self.size)
        return pil_img


def create_carla_rgb_transform(
    input_size, need_scale=True, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
):

    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size
    tfl = []

    if isinstance(input_size, (tuple, list)):
        input_size_num = input_size[-1]
    else:
        input_size_num = input_size

    if need_scale:
        if input_size_num == 112:
            tfl.append(Resize2FixedSize((170, 128)))
        elif input_size_num == 128:
            tfl.append(Resize2FixedSize((195, 146)))
        elif input_size_num == 224:
            tfl.append(Resize2FixedSize((341, 256)))
        elif input_size_num == 256:
            tfl.append(Resize2FixedSize((288, 288)))
        else:
            raise ValueError("Can't find proper crop size")
    tfl.append(transforms.CenterCrop(img_size))
    tfl.append(transforms.ToTensor())
    tfl.append(transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)))

    return transforms.Compose(tfl)


class LMB2DAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        print("setup!")
        self._hic = DisplayInterface()
        self.track = autonomous_agent.Track.SENSORS
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False
        self.rgb_front_transform = create_carla_rgb_transform(224)
        self.rgb_left_right_transform = create_carla_rgb_transform(128)
        self.rgb_center_transform = create_carla_rgb_transform(128, need_scale=False)

        self.active_misleading_instruction = False
        self.remaining_misleading_frames = 0

        self.visual_feature_buffer = []

        if IS_BENCH2DRIVE:
            self.save_name = path_to_conf_file.split('+')[-1]
            self.config_path = path_to_conf_file.split('+')[0]
        else:
            print("error")
            # self.config_path = path_to_conf_file
            # self.save_name = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
   
        self.config = imp.load_source("MainModel", self.config_path).GlobalConfig()

        self.turn_controller = PIDController(K_P=self.config.turn_KP, K_I=self.config.turn_KI, K_D=self.config.turn_KD, n=self.config.turn_n)
        self.speed_controller = PIDController(K_P=self.config.speed_KP, K_I=self.config.speed_KI, K_D=self.config.speed_KD, n=self.config.speed_n)

        model_cls = registry.get_model_class('vicuna_drive')

        self.agent_use_notice = self.config.agent_use_notice
        self.traffic_light_notice = ''
        self.curr_notice = ''
        self.now_notice_frame_id = -1
        self.sample_rate = self.config.sample_rate * 2 # The frequency of CARLA simulation is 20Hz

        print('build model...')
        model = model_cls(preception_model=self.config.preception_model,
                          preception_model_ckpt=self.config.preception_model_ckpt,
                          llm_model=self.config.llm_model,
                          max_txt_len=64,
                          use_notice_prompt=self.config.agent_use_notice,
                          )
        self.net = model

        print('load model...')
        self.net.load_state_dict(torch.load(self.config.lmb2d_ckpt)["model"], strict=False)
        self.net.cuda()
        self.net.eval()
        self.softmax = torch.nn.Softmax(dim=1)
        self.prev_lidar = None
        self.prev_control = None
        self.curr_instruction = 'Drive safely.'
        self.sampled_scenarios = []
        self.instruction = ''

        self.save_path = None
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ["ROUTES"]).stem + "_"
            string += "_".join(
                map(
                    lambda x: "%02d" % x,
                    (now.month, now.day, now.hour, now.minute, now.second),
                )
            )
            string = self.save_name + string
            print("Data save path:", string)

            self.save_path = pathlib.Path(SAVE_PATH) / string
            self.save_path.mkdir(parents=True, exist_ok=False)
            (self.save_path / "rgb").mkdir(parents=True, exist_ok=False)
            (self.save_path / "lidar").mkdir(parents=True, exist_ok=False)

    def _init(self):
        print("_init!")
        self._route_planner = RoutePlanner(5, 50.0)
        self._route_planner.set_route(self._global_plan, True)
        print("self.scenario_cofing_name: ", self.scenario_cofing_name)
        self._instruction_planner = InstructionPlanner(self.scenario_cofing_name, True)
        self.initialized = True
        random.seed(''.join([str(x[0]) for x in self._global_plan]))

    def _get_position(self, tick_data):
        gps = tick_data["gps"]
        gps = (gps - self._route_planner.mean) * self._route_planner.scale
        return gps

    def sensors(self):
        return [
            {
                "type": "sensor.camera.rgb",
                "x": 1.3,
                "y": 0.0,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "width": 1600,
                "height": 900,
                "fov": 70,
                "id": "rgb_front",  # 前
            },
            {
                "type": "sensor.camera.rgb",
                "x": 1.3,
                "y": 0.0,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -60.0,
                "width": 1600,
                "height": 900,
                "fov": 70,
                "id": "rgb_front_left",  # 左前
            },
            {
                "type": "sensor.camera.rgb",
                "x": 1.3,
                "y": 0.0,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 60.0,
                "width": 1600,
                "height": 900,
                "fov": 70,
                "id": "rgb_front_right",  # 右前
            },
            {
                "type": "sensor.camera.rgb",
                "x": -1.3,
                "y": 0.0,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 180.0,
                "width": 1600,
                "height": 900,
                "fov": 70,
                "id": "rgb_back",  # 后
            },
            {
                "type": "sensor.camera.rgb",
                "x": 1.3,
                "y": 0.0,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -120.0,
                "width": 1600,
                "height": 900,
                "fov": 70,
                "id": "rgb_back_left",  # 左后
            },
            {
                "type": "sensor.camera.rgb",
                "x": 1.3,
                "y": 0.0,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 120.0,
                "width": 1600,
                "height": 900,
                "fov": 70,
                "id": "rgb_back_right",  # 右后
            },
            {
                "type": "sensor.lidar.ray_cast",
                "x": 1.3,
                "y": 0.0,
                "z": 2.5,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -90.0,
                "id": "lidar", # 雷达
            },
            {
                "type": "sensor.other.imu",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.05,
                "id": "imu",
            },
            {
                "type": "sensor.other.gnss",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.01,
                "id": "gps",
            },
            {"type": "sensor.speedometer", "reading_frequency": 20, "id": "speed"},
        ]

    def tick(self, input_data):

        rgb_front = cv2.cvtColor(input_data["rgb_front"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_front_left = cv2.cvtColor(input_data["rgb_front_left"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_front_right = cv2.cvtColor(input_data["rgb_front_right"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_back_left = cv2.cvtColor(input_data["rgb_back_left"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_back_right = cv2.cvtColor(input_data["rgb_back_right"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_back = cv2.cvtColor(input_data["rgb_back"][1][:, :, :3], cv2.COLOR_BGR2RGB)

        gps = input_data["gps"][1][:2]
        speed = input_data["speed"][1]["speed"]
        compass = input_data["imu"][1][-1]
        if (
            math.isnan(compass) == True
        ):  # It can happen that the compass sends nan for a few frames
            compass = 0.0

        result = {
            "rgb_front": rgb_front,
            "rgb_front_left": rgb_front_left,
            "rgb_front_right": rgb_front_right,
            "rgb_back_left": rgb_back_left,
            "rgb_back_right": rgb_back_right,
            'rgb_back': rgb_back,
            "gps": gps,
            "speed": speed,
            "compass": compass,
        }

        pos = self._get_position(result)

        lidar_data = input_data['lidar'][1] # len(input_data['lidar'])=2, 第0位是一个数字
        result['raw_lidar'] = lidar_data[..., :3]

        lidar_unprocessed = lidar_data[..., :3]
        if self.prev_lidar is not None:
            lidar_unprocessed_full = np.concatenate([lidar_unprocessed, self.prev_lidar])
        else:
            lidar_unprocessed_full = lidar_unprocessed
        self.prev_lidar = lidar_unprocessed

        lidar_processed, num_points= lidar_to_raw_features(lidar_unprocessed_full)
        result['lidar'] = lidar_processed
        result['num_points'] = num_points

        result["gps"] = pos
        next_wp, next_cmd = self._route_planner.run_step(pos)
        result["next_waypoint"] = next_wp
        result["next_command"] = next_cmd.value
        result['measurements'] = [pos[0], pos[1], compass, speed]
        result['speed'] = speed

        theta = compass + np.pi / 2
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        local_command_point = np.array([next_wp[0] - pos[0], next_wp[1] - pos[1]])
        local_command_point = R.T.dot(local_command_point)
        result["target_point"] = local_command_point

        return result


    def update_and_collect(self, image_embeds):
        if 'lane' in self.curr_instruction: # change lane
            sample_rate = 1
        else:
            sample_rate = 2
        sample_rate = 2
        self.visual_feature_buffer.append(image_embeds)
        result = self.visual_feature_buffer[::self.sample_rate]
        if (len(self.visual_feature_buffer) -1) % self.sample_rate != 0:
            result.append(self.visual_feature_buffer[-1])
        return torch.stack(result, 1)

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        self.step += 1

        tick_data = self.tick(input_data)

        if self.step < 20:
            control = carla.VehicleControl()
            control.steer = float(0)
            control.throttle = float(0)
            control.brake = float(1)
            return control

        if self.step % 2 != 0 and self.step > 4:
            return self.prev_control

        velocity = tick_data["speed"]
        command = tick_data["next_command"]

        rgb_front = (
            self.rgb_front_transform(Image.fromarray(tick_data["rgb_front"]))
            .unsqueeze(0)
            .cuda()
            .float()
        )
        rgb_front_left = (
            self.rgb_left_right_transform(Image.fromarray(tick_data["rgb_front_left"]))
            .unsqueeze(0)
            .cuda()
            .float()
        )
        rgb_front_right = (
            self.rgb_left_right_transform(Image.fromarray(tick_data["rgb_front_right"]))
            .unsqueeze(0)
            .cuda()
            .float()
        )
        rgb_back_left = (
            self.rgb_left_right_transform(Image.fromarray(tick_data["rgb_back_left"]))
            .unsqueeze(0)
            .cuda()
            .float()
        )
        rgb_back_right = (
            self.rgb_left_right_transform(Image.fromarray(tick_data["rgb_back_right"]))
            .unsqueeze(0)
            .cuda()
            .float()
        )
        rgb_back = (
            self.rgb_left_right_transform(Image.fromarray(tick_data["rgb_back"]))
            .unsqueeze(0)
            .cuda()
            .float()
        )
        rgb_center = (
            self.rgb_center_transform(Image.fromarray(cv2.resize(tick_data["rgb_front"], (960, 540))))
            .unsqueeze(0)
            .cuda()
            .float()
        )

        last_instruction = self._instruction_planner.command2instruct(self.town_id, tick_data, self._route_planner.route)
        last_notice = self._instruction_planner.pos2notice(self.sampled_scenarios, tick_data)
        last_traffic_light_notice = self._instruction_planner.traffic_notice(tick_data)
        last_misleading_instruction = self._instruction_planner.command2mislead(self.town_id, tick_data)

        if last_notice == '':
            last_notice = last_traffic_light_notice

        if self.curr_instruction != last_instruction or len(self.visual_feature_buffer) > 400:
            if self.remaining_misleading_frames > 0:
                self.remaining_misleading_frames = self.remaining_misleading_frames - 1
            else:
                self.active_misleading_instruction = False
                if last_misleading_instruction!= '' and random.random() < 0.2:
                    self.curr_instruction = last_misleading_instruction
                    self.active_misleading_instruction = True
                    self.remaining_misleading_frames = 20
                else:
                    self.curr_instruction = last_instruction
                self.visual_feature_buffer = []
                self.curr_notice = ''
                self.curr_notice_frame_id = -1


        input_data = {}
        input_data["rgb_front"] = rgb_front
        input_data["rgb_front_left"] = rgb_front_left
        input_data["rgb_front_right"] = rgb_front_right
        input_data["rgb_back_left"] = rgb_back_left
        input_data["rgb_back_right"] = rgb_back_right
        input_data["rgb_center"] = rgb_center
        input_data["rgb_back"] = rgb_back
        input_data['target_point'] = torch.tensor(tick_data['target_point']).cuda().view(1,2).float()
        input_data["lidar"] = (torch.from_numpy(tick_data["lidar"]).float().cuda().unsqueeze(0))
        input_data['num_points'] = torch.tensor([tick_data['num_points']]).cuda().unsqueeze(0)
        input_data['velocity'] = torch.tensor([tick_data['speed']]).cuda().view(1, 1).float()
        input_data['text_input'] = [self.curr_instruction]
        image_embeds = self.net.visual_encoder(input_data) # 视觉模型
        image_embeds = self.update_and_collect(image_embeds)
        input_data['valid_frames'] = [image_embeds.size(1)]

        if last_notice != '' and last_notice != self.curr_notice:
            new_notice_flag = True
            self.curr_notice = last_notice
            self.curr_notice_frame_id = image_embeds.size(1) - 1
        else:
            new_notice_flag = False

        if self.agent_use_notice:
            input_data['notice_text'] = [self.curr_notice]
            input_data['notice_frame_id'] = [self.curr_notice_frame_id]

        with torch.cuda.amp.autocast(enabled=True):
            waypoints, is_end = self.net(input_data, inference_mode=True, image_embeds=image_embeds)

        waypoints = waypoints[-1]
        waypoints = waypoints.view(5, 2)
        end_prob = self.softmax(is_end)[-1][1] # is_end[1] means the prob of the frame is the last frame

        steer, throttle, brake, metadata = self.control_pid(waypoints, velocity)

        if end_prob > 0.75:
            self.visual_feature_buffer = []
            self.curr_notice = ''
            self.curr_notice_frame_id = -1

        if brake < 0.05:
            brake = 0.0
        if brake > 0.1:
            throttle = 0.0

        control = carla.VehicleControl()
        control.steer = float(steer) * 0.8
        control.throttle = float(throttle)
        control.brake = float(brake)

        display_data = {}
        display_data['rgb_front']       = cv2.resize(tick_data['rgb_front'], (1200, 900))
        display_data['rgb_front_left']  = cv2.resize(tick_data['rgb_front_left'], (280, 210))
        display_data['rgb_front_right'] = cv2.resize(tick_data['rgb_front_right'], (280, 210))
        display_data['rgb_back_left']   = cv2.resize(tick_data['rgb_back_left'], (280, 210))
        display_data['rgb_back_right']  = cv2.resize(tick_data['rgb_back_right'], (280, 210))
        display_data['rgb_center']      = cv2.resize(tick_data['rgb_front'][330:570, 480:720], (210, 210))
        if self.active_misleading_instruction:
            display_data['instruction'] = "Instruction: [Misleading] %s" % input_data['text_input'][0]
        else:
            display_data['instruction'] = "Instruction: %s" % input_data['text_input'][0]
        display_data['time'] = 'Time: %.3f. Frames: %d. End prob: %.2f' % (timestamp, len(self.visual_feature_buffer), end_prob)
        display_data['meta_control'] = 'Throttle: %.2f. Steer: %.2f. Brake: %.2f' %(
            control.steer, control.throttle, control.brake
        )
        display_data['waypoints'] = 'Waypoints: (%.1f, %.1f), (%.1f, %.1f)' % (waypoints[0,0], -waypoints[0,1], waypoints[1,0], -waypoints[1,1])
        display_data['notice'] = "Notice: %s" % last_notice
        surface = self._hic.run_interface(display_data)
        tick_data['surface'] = surface

        if self.step % 2 != 0 and self.step > 4:
            control = self.prev_control
        else:
            self.prev_control = control

        if SAVE_PATH is not None:
            self.save(tick_data)

        return control

    # ! 保存lidar的bev视角图
    def lidar_to_bev(self, lidar, min_x=-100,max_x=100, min_y=-100, max_y=100, pixels_per_meter=4, hist_max_per_pixel=2):
        xbins = np.linspace(min_x, max_x+1, (max_x - min_x) * pixels_per_meter + 1)
        ybins = np.linspace(min_y, max_y+1, (max_y - min_y) * pixels_per_meter + 1)

        # Compute histogram of x and y coordinates of points.
        hist = np.histogramdd(lidar[..., :2], bins=(xbins, ybins))[0]
        # Clip histogram
        hist[hist > hist_max_per_pixel] = hist_max_per_pixel
        # Normalize histogram by the maximum number of points in a bin we care about.
        overhead_splat = hist / hist_max_per_pixel * 255.
        # Return splat in X x Y orientation, with X parallel to car axis, Y perp, both parallel to ground.
        return overhead_splat[::-1,:]

    def save(self, tick_data):
        frame = (self.step - 20)
        Image.fromarray(tick_data["surface"]).save(
            self.save_path / "rgb" / ("%05d.jpg" % frame)
        )

        # print("lidar:", tick_data["lidar"])
        
        # ! 保存lidar的bev视角图
        # lidar_transformed = deepcopy(tick_data['lidar']) 
        # lidar_transformed[:, 1] *= -1
        # lidar_transformed = torch.from_numpy(self.lidar_to_histogram_features(lidar_transformed))
        # lidar_transformed_degrees = [lidar_transformed]
        # lidar_bev = torch.cat(lidar_transformed_degrees[::-1], dim=1)
        # lidar_bev = np.array(lidar_bev)
        # print("lidar_bev:", lidar_bev.shape)
        
        lidar_viz = self.lidar_to_bev(tick_data["lidar"]).astype(np.uint8)
        lidar_viz = cv2.cvtColor(lidar_viz, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(os.path.join(self.save_path, "lidar", "lidar_{:05d}.jpg".format(frame)), lidar_viz)

    def destroy(self):
        del self.net

    def control_pid(self, waypoints, velocity):
        '''
        Predicts vehicle control with a PID controller.
        Args:
            waypoints (tensor): predicted waypoints
            velocity (tensor): speedometer input
        '''
        assert(waypoints.size(0)==5)
        waypoints = waypoints.data.cpu().numpy()

        # flip y is (forward is negative in our waypoints)
        waypoints[:,1] *= -1
        speed = velocity

        desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0
        brake = desired_speed < self.config.brake_speed or (speed / desired_speed) > self.config.brake_ratio

        aim = (waypoints[1] + waypoints[0]) / 2.0
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        if(speed < 0.01):
            angle = np.array(0.0) # When we don't move we don't want the angle error to accumulate in the integral
        steer = self.turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.config.max_throttle)
        throttle = throttle if not brake else 0.0

        metadata = {
            'speed': float(speed.astype(np.float64)),
            'steer': float(steer),
            'throttle': float(throttle),
            'brake': float(brake),
            'wp_2': tuple(waypoints[1].astype(np.float64)),
            'wp_1': tuple(waypoints[0].astype(np.float64)),
            'desired_speed': float(desired_speed.astype(np.float64)),
            'angle': float(angle.astype(np.float64)),
            'aim': tuple(aim.astype(np.float64)),
            'delta': float(delta.astype(np.float64)),
        }

        return steer, throttle, brake, metadata
