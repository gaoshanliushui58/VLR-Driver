import json
import glob
import argparse
import os

def merge_route_json(folder_path):
    file_paths = glob.glob(f'{folder_path}/*.json')
    # print(file_paths)
    merged_records = []
    driving_score = []
    success_num = 0
    failed_scenario_info = []
    for file_path in file_paths:
        if 'merged.json' in file_path: continue
        if 'failed.json' in file_path: continue
        with open(file_path) as file:
            data = json.load(file)
            records = data['_checkpoint']['records']
            for rd in records:
                rd.pop('index')
                merged_records.append(rd)
                driving_score.append(rd['scores']['score_composed'])
                if rd['status']=='Completed' or rd['status']=='Perfect':
                    success_flag = True
                    for k,v in rd['infractions'].items():
                        if len(v)>0 and k != 'min_speed_infractions':
                            success_flag = False
                            failed_scenario_info.append((rd['route_id'], rd['scenario_name'], rd['weather_id'], rd['town_name'], rd['num_infractions'], 
                                                         rd['scores']['score_route'], rd['scores']['score_penalty'], rd['scores']['score_composed']))
                            print(rd['route_id'], "failed") #
                            break
                    if success_flag:
                        success_num += 1
                        print(rd['route_id'], "success") #
                else:
                    failed_scenario_info.append((rd['route_id'], rd['scenario_name'], rd['weather_id'], rd['town_name'], rd['num_infractions'], 
                                                 rd['scores']['score_route'], rd['scores']['score_penalty'], rd['scores']['score_composed']))
                    print(rd['route_id'], "failed") #
    if len(merged_records) != 220:
        print(f"-----------------------Warning: there are {len(merged_records)} routes in your json, which does not equal to 220. All metrics (Driving Score, Success Rate, Ability) are inaccurate!!!")
    merged_records = sorted(merged_records, key=lambda d: d['route_id'], reverse=True)
    _checkpoint = {
        "records": merged_records
    }

    merged_data = {
        "_checkpoint": _checkpoint,
        "driving score": sum(driving_score) / len(merged_records), # 220,
        "success rate": success_num / len(merged_records), # 220,
        "eval num": len(driving_score),
    }
    failed_detail_json(folder_path, failed_scenario_info)
    
    with open(os.path.join(folder_path, 'merged.json'), 'w') as file:
        json.dump(merged_data, file, indent=4)


def failed_detail_json(folder_path, failed_scenario_info):
    with open(os.path.join(folder_path, 'failed.json'), 'w') as file:
        file.write(f"id     scenario\t\t\t\t\t\t weat  town  infrac  RC\t     IS\t   DS\n")
        for route_id, scenario_name, weather_id, town_name, num_infractions, score_route, score_penalty, score_composed in failed_scenario_info:
            route_id = route_id.split('_')[1]
            scenario_name = scenario_name.split('_')[0]
            town_name = town_name.split('n')[1]
            file.write(f"{route_id:<7}{scenario_name:<30}{weather_id:<6}{town_name:<6}{num_infractions:<8}{score_route:<8}{score_penalty:<6}{score_composed:<8}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', help='old foo help')
    args = parser.parse_args()
    merge_route_json(args.folder)
