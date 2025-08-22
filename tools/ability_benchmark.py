import json
import carla
import argparse
import xml.etree.ElementTree as ET
from agents.navigation.global_route_planner import GlobalRoutePlanner
import os
import atexit
import subprocess
import time
import random

Ability = {
    "Overtaking":['Accident', 'AccidentTwoWays', 'ConstructionObstacle', 'ConstructionObstacleTwoWays', 'HazardAtSideLaneTwoWays', 'HazardAtSideLane', 'ParkedObstacleTwoWays', 'ParkedObstacle', 'VehicleOpenDoorTwoWays'],
    "Merging": ['CrossingBicycleFlow', 'EnterActorFlow', 'HighwayExit', 'InterurbanActorFlow', 'HighwayCutIn', 'InterurbanAdvancedActorFlow', 'MergerIntoSlowTrafficV2', 'MergeIntoSlowTraffic', 'NonSignalizedJunctionLeftTurn', 'NonSignalizedJunctionRightTurn', 'NonSignalizedJunctionLeftTurnEnterFlow', 'ParkingExit', 'SequentialLaneChange', 'SignalizedJunctionLeftTurn', 'SignalizedJunctionRightTurn', 'SignalizedJunctionLeftTurnEnterFlow'],
    "Emergency_Brake": ['BlockedIntersection', 'DynamicObjectCrossing', 'HardBreakRoute', 'OppositeVehicleTakingPriority', 'OppositeVehicleRunningRedLight', 'ParkingCutIn', 'PedestrianCrossing', 'ParkingCrossingPedestrain', 'StaticCutIn', 'VehicleTurningRoute', 'VehicleTurningRoutePedestrian', 'ControlLoss'],
    "Give_Way": ['InvadingTurn', 'YieldToEmergencyVehicle'],
    "Traffic_Signs": ['BlockedIntersection', 'OppositeVehicleTakingPriority', 'OppositeVehicleRunningRedLight', 'PedestrianCrossing', 'VehicleTurningRoute', 'VehicleTurningRoutePedestrian', 'EnterActorFlow', 'CrossingBicycleFlow', 'NonSignalizedJunctionLeftTurn', 'NonSignalizedJunctionRightTurn', 'NonSignalizedJunctionLeftTurnEnterFlow', 'OppositeVehicleTakingPriority', 'OppositeVehicleRunningRedLight', 'PedestrianCrossing', 'SignalizedJunctionLeftTurn', 'SignalizedJunctionRightTurn', 'SignalizedJunctionLeftTurnEnterFlow', 'T_Junction', 'VanillaNonSignalizedTurn', 'VanillaSignalizedTurnEncounterGreenLight', 'VanillaSignalizedTurnEncounterRedLight', 'VanillaNonSignalizedTurnEncouterStopsign', 'VehicleTurningRoute', 'VehicleTurningRoutePedestrian']
}

def get_infraction_status(record):
    for infraction,  value in record['infractions'].items():
        if infraction == "min_speed_infractions":
            continue
        elif len(value) > 0: # 车辆有违规行为
            return True
    return False

def update_Ability(scenario_name, Ability_Statistic, status):
    for ability, scenarios in Ability.items():
        if scenario_name in scenarios:
            Ability_Statistic[ability][1] += 1 # 总数+1
            if status: # 如果是complete并且没有违规，表示成功
                Ability_Statistic[ability][0] += 1
    pass

def update_Success(scenario_name, Success_Statistic, status):
    if scenario_name not in Success_Statistic: # 第一次遇到这个场景，要初始化
        if status:
            Success_Statistic[scenario_name] = [1, 1] # 成功数、总数
        else:
            Success_Statistic[scenario_name] = [0, 1]
    else: # 之前已经有这个场景，直接+1
        Success_Statistic[scenario_name][1] += 1
        if status:
            Success_Statistic[scenario_name][0] += 1
    pass

def get_position(xml_route):
    waypoints_elem = xml_route.find('waypoints')
    keypoints = waypoints_elem.findall('position')
    return [carla.Location(float(pos.get('x')), float(pos.get('y')), float(pos.get('z'))) for pos in keypoints]

def get_route_result(records, route_id):
    for record in records:
        record_route_id = record['route_id'].split('_')[1]
        if route_id == record_route_id:
            return record
    return None

def get_waypoint_route(locs, grp):
    route = []
    for i in range(len(locs) - 1):
        loc = locs[i]
        loc_next = locs[i + 1]
        interpolated_trace = grp.trace_route(loc, loc_next)
        for wp, _ in interpolated_trace:
            route.append(wp)
    return route

def main(args):
    routes_file = args.file 
    result_file = args.result_file
    Ability_Statistic = {}
    crash_route_list = []
    for key in Ability:
        Ability_Statistic[key] = [0, 0] # 五项能力：Overtaking  Merging  Emergency_Brake  Give_Way  Traffic_Signs
    # print("Ability_Statistic:", Ability_Statistic)
    Success_Statistic = {}
    
    with open(result_file, 'r') as f:
        data = json.load(f)
    records = data["_checkpoint"]["records"]
    
    tree = ET.parse(routes_file)
    # print("tree:", tree)
    root = tree.getroot()
    # print("root:", root)
    routes = root.findall('route') # n条route路线
    # print("routes:", routes)
    sorted_routes = sorted(routes, key=lambda x: x.get('town')) # 按town序号排序
    # print("sorted_routes:", sorted_routes)
    
    carla_path = os.environ["CARLA_ROOT"]
    # cmd1 = f"{os.path.join(carla_path, 'CarlaUE4.sh')} -RenderOffScreen -nosound -carla-rpc-port={args.port}"
    # server = subprocess.Popen(cmd1, shell=True, preexec_fn=os.setsid) # 开carla服务器
    # print(cmd1, server.returncode, flush=True)
    # time.sleep(10)
    client = carla.Client(args.host, args.port) # carla客户端
    client.set_timeout(300)
    
    current_town = sorted_routes[0].get('town')
    world = client.load_world(current_town)
    carla_map = world.get_map()
    grp = GlobalRoutePlanner(carla_map, 1.0)
    for route in sorted_routes:
        scenarios = route.find('scenarios')
        scenario_name = scenarios.find('scenario').get("type") # 'ParkingCutIn'等
        route_id = route.get('id')
        route_record = get_route_result(records, route_id) # 返回merge中的结果
        if route_record is None:
            crash_route_list.append((scenario_name, route_id))
            print('No result record of route', route_id, "in the result file")
            continue
        if route_record["status"] == 'Completed' or route_record["status"] == "Perfect":
            if get_infraction_status(route_record): # 有违规
                record_success_status = False
            else:
                record_success_status = True # 没违规
        else:
            record_success_status = False # 没完成failed
        update_Ability(scenario_name, Ability_Statistic, record_success_status) # 给五项能力计数（成功数、总数）
        update_Success(scenario_name, Success_Statistic, record_success_status) # 给每一项细分能力计数（成功数、总数）
        # if scenario_name in Ability["Traffic_Signs"] and (scenario_name in Ability["Merging"] or scenario_name in Ability["Emergency_Brake"]):
        # Only these three 'Ability's intersect
        if scenario_name in Ability["Traffic_Signs"]:
            # Only these three 'Ability's intersect
            if route.get('town') != current_town:
                current_town = route.get('town')
                print("Loading the town:", current_town)
                world = client.load_world(current_town)
                print("successfully load the town:", current_town)
            carla_map = world.get_map()
            grp = GlobalRoutePlanner(carla_map, 1.0)
            location_list = get_position(route)
            waypoint_route = get_waypoint_route(location_list, grp)
            count = 0
            for wp in waypoint_route:
                count += 1
                if wp.is_junction:
                    break 
            if not wp.is_junction:
                raise RuntimeError("This route does not contain any junction-waypoint!")
            # +8 to ensure the ego pass the trigger volume
            junction_completion = float(count+8) / float(len(waypoint_route))
            record_completion = route_record["scores"]["score_route"] / 100.0
            stop_infraction = route_record["infractions"]["stop_infraction"]
            red_light_infraction = route_record["infractions"]["red_light"]
            if record_completion > junction_completion and not stop_infraction and not red_light_infraction:
                Ability_Statistic['Traffic_Signs'][0] += 1
                Ability_Statistic['Traffic_Signs'][1] += 1
            else:
                Ability_Statistic['Traffic_Signs'][1] += 1
        else:
            pass
    
    # 五项能力的成功率
    Ability_Res = {}
    Ability_Res_sum = 0
    Ability_Res_vaild_num = 0
    for ability, statis in Ability_Statistic.items():
        if statis[1] == 0:
            Ability_Res[ability] = -1 
        else:
            Ability_Res[ability] = float(statis[0])/float(statis[1])
            Ability_Res_sum += Ability_Res[ability]
            Ability_Res_vaild_num += 1
        
    for key, value in Ability_Res.items():
        print(key, ": ", value)
    
    Ability_Res['mean'] = float(Ability_Res_sum) / float(Ability_Res_vaild_num) # 五项能力成功率均值
    Ability_Res['crashed'] = crash_route_list

    # 细分能力的成功率
    Success_Res = {}
    Route_num = 0
    Succ_Route_num = 0
    for scenario, statis in Success_Statistic.items():
        Success_Res[scenario] = float(statis[0])/float(statis[1])
        Succ_Route_num += statis[0]
        Route_num += statis[1]
    
    with open(f"{result_file.split('.')[0]}_ability.json", 'w') as file:
        json.dump(Ability_Res, file, indent=4)
        json.dump(Success_Res, file, indent=4)
        json.dump(Ability_Statistic, file, indent=4)
        json.dump(Success_Statistic, file, indent=4)


    # assert len(crash_route_list) == 220 - float(Route_num)
    if len(crash_route_list) != 220 - float(Route_num):
        print(f"-----------------------Warning: there are {Route_num} routes in your json, which does not equal to 220.")
    print(f'Crashed Route num: {len(crash_route_list)}, Crashed Route ID: {crash_route_list}')
    print('Finished!')

if __name__=='__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-f', '--file', nargs=None, default="HOME/Bench2Drive/leaderboard/data/bench2drive5.xml", help='route file')
    argparser.add_argument('-r', '--result_file', nargs=None, default="HOME/Bench2Drive/output/tmp0925_5route_01/merged.json", help='result json file')
    argparser.add_argument('-t', '--host', default='localhost', help='IP of the host server (default: localhost)')
    argparser.add_argument('-p', '--port', nargs=1, default=2000, help='carla rpc port')
    args = argparser.parse_args()
    main(args)
    