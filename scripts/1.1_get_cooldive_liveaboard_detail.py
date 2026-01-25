import requests
import re
import os
import json
import time
import random
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

# --- 配置部分 ---
BASE_URL = "https://www.cooldive.com.cn"
PRODUCT_LIST_API = f"{BASE_URL}/api/v1/search/boat"
DETAIL_PAGE_URL_TEMPLATE = f"{BASE_URL}/boat/{{boat_id}}/detail"
TRIP_SCHEDULE_API = f"{BASE_URL}/api/v1/boat/trip"

# 用于记录已爬取船宿ID的文件
CRAWLED_IDS_FILE = "crawled_boat_ids.json"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Referer': f"{BASE_URL}/",
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'same-origin',
    'Upgrade-Insecure-Requests': '1',
}

# --- 反爬配置 ---
MAX_RETRIES = 3  # 最大重试次数
RETRY_DELAY = 10  # 重试前的等待时间（秒）

# # 使用你的VPN代理配置
# PROXIES = {
#     "http": "http://127.0.0.1:7897",
#     "https": "http://127.0.0.1:7897",
# }
# print(f"代理配置: {PROXIES}")

# --- 辅助函数：清理HTML标签 ---
def clean_html(html_text):
    if not isinstance(html_text, str):
        return html_text
    # 移除 HTML 标签，替换 <p> 为换行，并清理多余空格和换行
    soup = BeautifulSoup(html_text, 'html.parser')
    for br in soup.find_all('br'):
        br.replace_with('\n')  # 将 <br> 替换为换行
    for p in soup.find_all('p'):
        p.append('\n\n')  # 在每个 <p> 标签后添加两个换行

    clean_text = soup.get_text()
    clean_text = clean_text.replace('\r\n', '\n').replace('\n\n\n', '\n\n').strip()  # 替换多余换行
    return clean_text


# --- 辅助函数：处理请求 (带重试和代理) ---
def make_request_with_retries(method, url, headers, params=None, data=None, json_data=None, retries=MAX_RETRIES,
                              delay_factor=RETRY_DELAY):
    for i in range(retries + 1):
        try:
            with requests.Session() as session:
                session.headers.update(headers)
                response = session.request(method, url, params=params, data=data, json=json_data,
                                           # proxies=PROXIES,
                                           timeout=15)
                response.raise_for_status()
                return response
        except requests.exceptions.HTTPError as e:
            if e.response.status_code in [403, 429]:
                print(f"遇到反爬状态码 {e.response.status_code}，正在重试...")
            else:
                print(f"HTTP错误 {e.response.status_code}: {e}，正在重试...")
        except requests.exceptions.RequestException as e:
            print(f"请求失败: {e}，正在重试...")

        if i < retries:
            sleep_time = delay_factor * (2 ** i) + random.uniform(1, 3)
            print(f"第 {i + 1} 次重试失败，等待 {sleep_time:.2f} 秒后重试...")
            time.sleep(sleep_time)
        else:
            print(f"所有重试均失败，放弃请求: {url}")
            return None
    return None

# --- 新增辅助函数：将布尔值字典转换为自然语言描述列表 ---
def describe_features_from_dict(feature_dict, feature_map):
    """
    将包含1/0布尔值的字典，根据映射表转换为自然语言描述列表。
    :param feature_dict: 原始字典，如 cabin, food, onboard, diving。
    :param feature_map: 映射表，key是字典的字段名，value是对应的自然语言描述。
    :return: 自然语言描述的列表。
    """
    descriptions = []
    if not isinstance(feature_dict, dict):
        return descriptions

    for key, desc in feature_map.items():
        if feature_dict.get(key) == 1: # 值为1表示有此功能
            descriptions.append(desc)
    return descriptions

# --- 具体字段映射表 (可以根据你的需求调整描述) ---
CABIN_FEATURES_MAP = {
    "En-SuiteCabins": "带独立卫浴",
    "AirConditioning": "有空调",
    "WallFan": "有壁扇",
    "TVinCabins": "客舱有电视",
    "LockableStorage": "带保险柜",
    "Towels": "提供毛巾",
    "Bathrobe": "提供浴袍",
    "Hairdryer": "提供吹风机",
    "Toiletries": "提供洗漱用品"
}

FOOD_FEATURES_MAP = {
    "WesternFood": "提供西餐",
    "LocalFood": "提供当地美食",
    "DietaryRestrictions": "提供素食",
    "BuffetStyle": "自助餐形式",
    "AlcoholicBeveragesandSpirits": "提供酒精饮料",
    "PlateService": "提供点餐服务",
    "Hot&ColdSoftDrinks": "提供冷热软饮",
    "SnacksAllDay": "全天提供小吃"
}

ONBOARD_FEATURES_MAP = {
    "Non-Diver(Snorkeler)Friendly": "非潜水员/浮潜者友好",
    "ChildFriendly": "儿童友好",
    "Massage": "按摩服务",
    "SPA": "水疗服务",
    "HotTub/Jacuzzi": "热水浴缸/按摩浴缸",
    "OnboardKayaks": "船上皮划艇",
    "LandExcursions": "陆地短途旅行",
    "Fishing": "钓鱼活动",
    "Surfing": "冲浪活动",
    "Bar": "酒吧",
    "IndoorSaloon": "室内沙龙",
    "AirConditionedSaloon": "空调沙龙",
    "Audio&VideoEntertainment": "音视频娱乐",
    "OpenAirSaloon": "露天酒吧",
    "SunDeck": "遮阳甲板",
    "SunLoungers": "日光躺椅",
    "LaundryService": "洗衣服务"
}

DIVING_FEATURES_MAP = {
    "DINAdaptors": "提供DIN适配器",
    "RebreatherSupport": "循环呼吸机",
    "RinseHosts": "冲洗室",
    "TendersforDiving": "潜水照料员",
    "Compressors": "压缩机",
    "DiveDeck": "专用潜水甲板",
    "WarmWaterShowers": "热水淋浴",
    "PersonalStorageSpace": "个人储物空间",
    "TechDiving": "技术潜水支持"
}


# --- 持久化已爬取ID的辅助函数 ---
def load_crawled_ids(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                return set(json.load(f))
            except json.JSONDecodeError:
                print(f"警告: 无法解析 {file_path}，将从空集合开始。")
                return set()
    return set()

def save_crawled_ids(file_path, crawled_ids_set):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(list(crawled_ids_set), f, ensure_ascii=False, indent=4)


# --- 1. 获取船宿ID列表的函数 ---
def get_boat_ids_from_list(max_pages=None, batch_size=None):
    """
    从船宿列表API获取船宿ID。
    该函数现在专注于获取“未爬取过”的船宿ID。
    :param max_pages: 限制从列表页获取的最大页数，None表示获取所有页。
    :param batch_size: 每次运行希望获取多少个新的船宿ID，None表示获取所有未爬取的。
    :return: 待爬取的新船宿ID列表。
    """
    all_api_boat_ids = []  # 从API获取的所有ID
    crawled_ids = load_crawled_ids(CRAWLED_IDS_FILE)  # 加载已经爬过的ID
    new_boat_ids_to_crawl = []  # 最终要返回的、未爬过的新ID

    page = 1
    total_found = -1

    print(f"开始获取船宿ID列表 (最多 {max_pages if max_pages is not None else '所有'} 页)...")
    print(f"  已爬取ID数量: {len(crawled_ids)}")

    while total_found == -1 or len(all_api_boat_ids) < total_found:
        if max_pages is not None and max_pages > 0 and page > max_pages:
            print(f"达到最大列表页爬取限制 ({max_pages} 页)，停止获取ID。")
            break

        # 如果已经获取到足够多的新ID，提前停止
        if batch_size is not None and len(new_boat_ids_to_crawl) >= batch_size:
            print(f"已获取到 {batch_size} 个新船宿ID，停止从列表页获取。")
            break

        params = {
            "limit": 20,
            "page": page,
            "keyword": ""
        }

        response = make_request_with_retries("GET", PRODUCT_LIST_API, HEADERS, params=params)
        if response is None:
            print("获取ID列表请求最终失败，无法继续。")
            break

        try:
            data = response.json()
            if data.get('code') == 200 and 'data' in data and 'list' in data['data']:
                current_page_boats = data['data']['list']
                total_found = data['data'].get('total', 0)

                if not current_page_boats:
                    print(f"第 {page} 页没有数据，停止获取ID。")
                    break

                for boat in current_page_boats:
                    if isinstance(boat, dict) and 'boatId' in boat:
                        # 先添加到所有API ID列表
                        all_api_boat_ids.append(boat['boatId'])
                        # 检查是否已爬取，且新ID数量未达标
                        if boat['boatId'] not in crawled_ids and (
                                batch_size is None or len(new_boat_ids_to_crawl) < batch_size):
                            new_boat_ids_to_crawl.append(boat['boatId'])

                print(
                    f"  已从API获取 {len(all_api_boat_ids)} 个ID (当前页 {page}), 其中 {len(new_boat_ids_to_crawl)} 个是新ID。")
                page += 1
                time.sleep(random.uniform(3, 7))

            else:
                print(f"获取ID列表API返回错误或数据结构不符：{data}")
                break

        except json.JSONDecodeError:
            print(f"获取ID列表JSON解析失败: {response.text}")
            break
        except Exception as e:
            print(f"获取ID列表时发生未知错误: {e}")
            break

    return list(set(new_boat_ids_to_crawl))  # 确保返回的新ID也是唯一的


# --- 2. 爬取船宿详情页数据的函数 ---
def fetch_boat_detail(boat_id):
    detail_url = DETAIL_PAGE_URL_TEMPLATE.format(boat_id=boat_id)

    response = make_request_with_retries("GET", detail_url, HEADERS)
    if response is None:
        return None

    html_content = response.text

    nuxt_data_match = re.search(r'window\.__NUXT__\s*=\s*(\{.+?\});', html_content, re.DOTALL)

    if nuxt_data_match:
        json_str = nuxt_data_match.group(1)
        json_str = json_str.replace('\\u002F', '/')

        try:
            nuxt_data = json.loads(json_str)
            detail_info = nuxt_data.get('data', [{}])[0].get('detailInfo', {})

            if not detail_info:
                print(f"警告: 船宿ID {boat_id} 的详情数据结构不符，可能没有detailInfo。")
                return None

            boat_info = detail_info.get('boatInfo', {})
            extra_info = boat_info.get('extraInfo', {})

            extracted_data = {
                "boatId": boat_info.get('id'),
                "rating": boat_info.get('rating'),
                "nameCN": boat_info.get('nameCN'),
                "nameEN": boat_info.get('nameEN'),
                "locationName": boat_info.get('locationName'),
                "yearBuilt": boat_info.get('yearBuilt'),
                "material": boat_info.get('material'),
                "length": boat_info.get('length'),
                "width": boat_info.get('width'),
                "nitrox": boat_info.get('nitrox'),
                "wifi": boat_info.get('wifi'),
                "diving_equipment": boat_info.get('diving_equipment'),
                "tech_diving_friendly": boat_info.get('tech_diving_friendly'),
                "languages": boat_info.get('languages'),
                "roomNum": boat_info.get('roomNum'),
                "descr": boat_info.get('descr'),
                "policy": boat_info.get('policy'),
                "included_cn": boat_info.get('included_cn'),
                "notincluded_cn": boat_info.get('notincluded_cn'),
                "payment": extra_info.get('boat', {}).get('payment'),
                "cabin": extra_info.get('cabin', {}),
                "food": extra_info.get('food', {}),
                "onboard": extra_info.get('onboard', {}),
                "updatedTime": boat_info.get('updateTime'),
                "diving": extra_info.get('diving', {}),
                "gallery": boat_info.get('gallery', []),
            }
            # --- 数据后处理：清理换行符和HTML标签 ---
            for key in ['descr', 'included_cn', 'notincluded_cn']:
                if isinstance(extracted_data.get(key), str):
                    extracted_data[key] = clean_html(extracted_data[key])

            if isinstance(extracted_data.get('policy'), list):
                extracted_data['policy'] = [clean_html(p) for p in extracted_data['policy'] if isinstance(p, str)]
            elif isinstance(extracted_data.get('policy'), str):
                extracted_data['policy'] = clean_html(extracted_data['policy'])

            return extracted_data

        except json.JSONDecodeError:
            print(f"解析船宿ID {boat_id} 详情页的JSON数据失败。JSON片段: {json_str[:500]}...")
            return None
        except Exception as e:
            print(f"处理船宿ID {boat_id} 详情时发生未知错误: {e}")
            return None

    else:
        print(f"警告: 未能在船宿ID {boat_id} 的详情页HTML中找到 window.__NUXT__ 数据。")
        return None


# --- 3. 新增：爬取船宿行程信息的函数 ---
def fetch_trip_schedules(boat_id, max_trips=None):
    """
    爬取单个船宿的所有行程班次信息（不包含实时价格和空位）。
    :param boat_id: 船宿ID。
    :param max_trips: 限制获取的行程数量。None表示获取所有。
    :return: 该船宿的行程列表。
    """
    all_trips = []
    cursor = 0  # 初始游标
    limit = 10  # 每页获取10条行程
    has_more = True  # 是否还有更多数据

    print(f"  正在获取船宿ID {boat_id} 的行程信息 (目标 {max_trips if max_trips is not None else '所有'} 条)...")

    while has_more:
        if max_trips is not None and len(all_trips) >= max_trips:
            print(f"    已获取目标数量的行程 ({max_trips} 条)，停止获取。")
            break

        params = {
            "limit": limit,
            "cursor": cursor,
            "startDate": "",  # 暂时不设置日期筛选，获取所有可用行程
            "endDate": "",
            "boatId": boat_id
        }

        # HEADERS调整为API请求的Accept类型
        api_headers = HEADERS.copy()
        api_headers['Accept'] = 'application/json, text/plain, */*'
        api_headers['Referer'] = f"{BASE_URL}/boat/{boat_id}/detail"  # 模仿从详情页发出的请求

        response = make_request_with_retries("GET", TRIP_SCHEDULE_API, api_headers, params=params)
        if response is None:
            print(f"  获取船宿ID {boat_id} 的行程请求最终失败。")
            break

        try:
            data = response.json()
            if data.get('code') == 200 and 'data' in data and 'list' in data['data']:
                current_page_trips = data['data']['list']
                next_cursor = data['data'].get('cursor')  # 获取下一页的游标
                has_more = data['data'].get('hasMore', False)  # 是否还有更多数据

                if not current_page_trips:
                    print(f"  船宿ID {boat_id} 游标 {cursor} 处没有更多行程数据。")
                    break

                for trip in current_page_trips:
                    if not isinstance(trip, dict):
                        print(f"警告: 船宿ID {boat_id} 发现非字典类型的行程数据，跳过: {trip}")
                        continue

                    # --- 提取和清理字段 ---
                    extracted_trip = {
                        "tripId": trip.get('tripId'), # 旅程ID
                        "boatId": trip.get('boatId'), # 船的ID
                        "tourId": trip.get('tourId'), # 路线的ID，可重复
                        "arrivalDate": trip.get('arrivalDate'), # 出发日期(时间戳）
                        "arrivalDate_display":datetime.fromtimestamp(trip.get('arrivalDate')).strftime("%Y-%m-%d"), # 出发日期(字符串版）
                        "departureDate": trip.get('departureDate'), # 返程抵达日期(时间戳）
                        "departureDate_display": datetime.fromtimestamp(trip.get('departureDate')).strftime(
                            "%Y-%m-%d"),  # 返程抵达日期(字符串版）
                        "nameCN": trip.get('nameCN'), # 路线名称
                        "nameEN": trip.get('nameEN'), # 路线名称（英文）

                        # 实时查询字段，这里仅作记录，不作为知识库内容
                        "availableCount_raw": trip.get('availableCount'), # 本船在该日期路线的空位总数（包含各种房型）
                        "price_raw": trip.get('price'), # 本船在该日期路线的最低价（是字典，包含数值和货币单位）
                        "priceOld_raw": trip.get('priceOld'), # 不清楚是什么，似乎没用

                        "policy": clean_html(trip.get('policy')),  # 预定和退款政策；policy字段是JSON字符串，需要额外处理
                    }

                    # 解析 policy 字段，它是一个JSON字符串，包含booking和cancellation policy
                    if isinstance(trip.get('policy'), str):
                        try:
                            policy_json = json.loads(trip['policy'])
                            extracted_trip['booking_policy'] = next(
                                (clean_html('; '.join(p['items'])) for p in policy_json if
                                 p.get('title') == 'Booking policy'), '')
                            extracted_trip['cancellation_policy'] = next(
                                (clean_html('; '.join(p['items'])) for p in policy_json if
                                 p.get('title') == 'Cancellation policy'), '')
                        except json.JSONDecodeError:
                            extracted_trip['booking_policy'] = extracted_trip['cancellation_policy'] = clean_html(
                                trip['policy'])  # 解析失败则用原始字符串
                    else:
                        extracted_trip['booking_policy'] = extracted_trip['cancellation_policy'] = ''

                    # --- 提取 tour 字段 ---
                    tour_data = trip.get('tour', {})
                    if tour_data:
                        extracted_trip['tour_details'] = {
                            "arrivalLocation": tour_data.get('arrivalLocation'), # 出发地点，通常是机场通过车转运到港口
                            "departureLocation": tour_data.get('departureLocation'), # 返程抵达地点
                            "arrivalPort": tour_data.get('arrivalPort'), # 出发港口
                            "departurePort": tour_data.get('departurePort'), # 返程抵达港口，通常是通过车转运到机场
                            "diveNumDesc": tour_data.get('diveNumDesc'), # 潜水次数
                            "photos": tour_data.get('photos', []), # 图片
                            "experience": tour_data.get('experience'), # 潜水次数（经验）要求
                            "certification": tour_data.get('sertification'), # 潜水证书要求
                            "dives": tour_data.get('dives'), # 潜水次数
                            "duration": tour_data.get('duration'), # 旅程时间（白天的数量）
                            "nights": tour_data.get('nights'), # 旅程时间（夜晚的数量）
                            "included_cn": clean_html(tour_data.get('included_cn')), # 费用已包含的服务
                            "notincluded_cn": clean_html(tour_data.get('notincluded_cn')), # 费用不包含的服务
                            "programm": clean_html(tour_data.get('programm')),  # 行程介绍，HTML格式
                            "check_in": tour_data.get('check_in', '').replace('.', ':'), # 登船时间，24小时制
                            "check_out": tour_data.get('check_out', '').replace('.', ':'),# 下船时间，24小时制
                        }

                        # --- 提取 tour.tripRoute (潜点/路线节点) ---
                        route_nodes = []
                        if isinstance(tour_data.get('tripRoute'), list):
                            for node in tour_data['tripRoute']:
                                if not isinstance(node, dict): continue
                                route_node = {"type": node.get('type')}
                                map_data = node.get('map', {})
                                if map_data:
                                    # 提取潜点详细信息 (仅当type为divesite时，map字段才完整)
                                    if node.get('type') == 'divesite':
                                        extracted_map_data = {
                                            "id": map_data.get('id'),
                                            "name": map_data.get('name'),
                                            "descr": clean_html(map_data.get('descr')),
                                            "whatToSee": clean_html(map_data.get('whatToSee')),
                                            "tags": [t.get('name') for t in map_data.get('tags', []) if
                                                     isinstance(t, dict)],
                                            "diversLevel": map_data.get('diversLevel')
                                            # 可以根据需要添加其他map字段，如latitude, longitude, image, url, rating等
                                        }
                                        route_node['map_details'] = extracted_map_data
                                    else:  # departure/arrival等节点，map只有经纬度
                                        route_node['map_coords'] = {
                                            "latitude": map_data.get('latitude'),
                                            "longitude": map_data.get('longitude')
                                        }
                                route_nodes.append(route_node)
                        extracted_trip['tour_details']['route_nodes'] = route_nodes

                        # --- 提取 tour.mapData (所有潜点/港口列表) ---
                        # 这个和tripRoute.map有一些重复，但提供了所有潜点的汇总
                        all_map_items = []
                        if isinstance(tour_data.get('mapData'), list):
                            for map_type_item in tour_data['mapData']:
                                if isinstance(map_type_item, dict) and isinstance(map_type_item.get('items'), list):
                                    for item in map_type_item['items']:
                                        if isinstance(item, dict) and item.get('type') == 'divesites':  # 仅提取潜点
                                            all_map_items.append({
                                                "id": item.get('id'), # 潜点ID
                                                "name": item.get('name'), # 潜点名称
                                                "descr": clean_html(item.get('descr')), # 潜点描述
                                                "whatToSee": clean_html(item.get('whatToSee')), # 潜点可能看见什么
                                                "tags": [t.get('name') for t in item.get('tags', []) if
                                                         isinstance(t, dict)],  # 潜点的TAG
                                                "diversLevel": item.get('diversLevel') # 潜水水平要求
                                            })
                        extracted_trip['tour_details']['all_divesites_in_route'] = all_map_items

                    all_trips.append(extracted_trip)
                    # 如果已达到max_trips，则不再添加
                    if max_trips is not None and len(all_trips) >= max_trips:
                        break

                print(f"    已获取 {len(all_trips)} 条行程数据 (游标 {cursor})。")
                cursor = next_cursor  # 更新游标
                if not next_cursor:  # 如果没有下一页游标，则认为没有更多数据
                    has_more = False
                time.sleep(random.uniform(2, 5))

            else:
                print(f"  API返回错误或数据结构不符：{data}")
                break

        except json.JSONDecodeError:
            print(f"  JSON解析失败: {response.text}")
            break
        except Exception as e:
            print(f"  发生未知错误: {e}")
            break

    return all_trips


# --- 主函数 ---
def get_cooldive_liveaboard_detail(output_json_file="cooldive_boat_details.json", max_list_pages=None, batch_size=None,
                                   max_trips_per_boat=None):
    """
    主爬虫函数，实现增量爬取船宿详情和行程。
    :param output_json_file: 输出的JSON文件名。
    :param max_list_pages: 从列表页获取的最大页数，None表示不限制。
    :param batch_size: 每次运行希望获取多少个新的船宿ID，None表示获取所有未爬取的。
                       这是实现“每次爬几个新的船”的关键。
    :param max_trips_per_boat: 限制每个船宿获取的行程数量。None表示获取所有。
    """
    print("--- 开始增量爬取船宿详情数据 ---")

    # 1. 获取需要爬取的新船宿ID
    new_boat_ids_to_crawl = get_boat_ids_from_list(max_pages=max_list_pages, batch_size=batch_size)

    if not new_boat_ids_to_crawl:
        print("未找到新的船宿ID进行爬取，停止。")
        return

    print(f"\n成功获取到 {len(new_boat_ids_to_crawl)} 个新船宿ID。开始逐一爬取详情和行程...")

    all_boat_details_with_trips = []
    crawled_ids_set = load_crawled_ids(CRAWLED_IDS_FILE)  # 重新加载已爬ID，确保最新

    for i, boat_id in enumerate(new_boat_ids_to_crawl):
        print(f"\n({i + 1}/{len(new_boat_ids_to_crawl)}) 正在爬取船宿ID: {boat_id} 的基本详情...")
        detail_data = fetch_boat_detail(boat_id)

        if detail_data:
            trip_schedules = fetch_trip_schedules(boat_id, max_trips=max_trips_per_boat)
            detail_data['schedules'] = trip_schedules
            all_boat_details_with_trips.append(detail_data)
            crawled_ids_set.add(boat_id)  # 将这个boatId标记为已爬取

        time.sleep(random.uniform(15, 20))

    # 2. 保存更新后的已爬取ID列表
    save_crawled_ids(CRAWLED_IDS_FILE, crawled_ids_set)
    print(f"\n已更新已爬取ID列表到 '{CRAWLED_IDS_FILE}'。")

    # 3. 保存爬取到的详情和行程数据到JSON文件
    try:
        # 可以选择追加到现有文件，或者覆盖。这里选择覆盖，更简单。
        # 如果需要追加，需要读取旧文件，合并数据，再写入。
        with open(output_json_file, 'w', encoding='utf-8') as f:
            json.dump(all_boat_details_with_trips, f, ensure_ascii=False, indent=4)
        print(f"\n本次爬取到的船宿详情和行程数据已成功保存到 '{output_json_file}'")
        print(f"共爬取了 {len(all_boat_details_with_trips)} 个新船宿的详情和行程。")
    except Exception as e:
        print(f"保存数据到文件 '{output_json_file}' 失败: {e}")


# --- 数据处理模块：分块与元数据标注 ---

# def generate_chunk_id(content, metadata):
#     """生成Chunk的唯一ID和内容哈希。"""
#     unique_string = json.dumps({"page_content": content, "metadata": metadata}, sort_keys=True, ensure_ascii=False)
#     chunk_id = hashlib.sha256(unique_string.encode('utf-8')).hexdigest()
#     return chunk_id, hashlib.md5(content.encode('utf-8')).hexdigest()

# =======================================================
# 定义需要特殊处理的元数据字段及其目标类型 (与 ingest.py 保持一致)
# =======================================================
FORCE_STR_KEYS = {
    'boatId', 'tourId', 'tripId', 'certification',
    'nameCN', 'nameEN', 'locationName', 'diving_equipment', 'languages', 'policy', 'wifi', 'tech_diving_friendly',
    'arrivalLocation','departureLocation',
}
FORCE_FLOAT_KEYS = {'rating'}
FORCE_INT_KEYS = {'yearBuilt', 'dives', 'duration', 'nights', 'experience'}
BOOLEAN_KEYS = {}
DATE_TIME_STR_KEYS = {'arrivalDate', 'departureDate', 'updatedTime','arrivalDate_display', 'departureDate_display'}


def normalize_single_metadata_value(col_name: str, value):
    """
    辅助函数：对单个元数据值进行类型转换和规范化。
    """
    if pd.isna(value) or value == '':
        return None

    str_value = str(value).strip()

    if col_name in FORCE_STR_KEYS:
        return str_value
    elif col_name in FORCE_FLOAT_KEYS:
        try:
            return float(str_value)
        except ValueError:
            print(
                f"警告 (1.1_get_cooldive_liveaboard_detail.py): '{col_name}' 值 '{str_value}' 无法转换为浮点数，设为 None。")
            return None
    elif col_name in FORCE_INT_KEYS:
        try:
            return int(float(str_value))
        except ValueError:
            print(
                f"警告 (1.1_get_cooldive_liveaboard_detail.py): '{col_name}' 值 '{str_value}' 无法转换为整数，设为 None。")
            return None
    elif col_name in BOOLEAN_KEYS:
        lower_val = str_value.lower()
        if lower_val == 'true' or lower_val == 'yes':
            return True
        elif lower_val == 'false' or lower_val == 'no':
            return False
        else:
            print(
                f"警告 (1.1_get_cooldive_liveaboard_detail.py): '{col_name}' 值 '{str_value}' 无法转换为布尔值，设为 None。")
            return None
    elif col_name in DATE_TIME_STR_KEYS:
            # 1. 优先尝试转为浮点数 (针对 arrivalDate, departureDate)
        try:
            return float(str_value)
        except ValueError:
            # 2. 转数字失败，尝试转日期对象 (针对 updatedTime, *_display)
            try:
                dt_obj = pd.to_datetime(str_value)

                if "_display" in col_name:
                    return dt_obj.strftime("%Y-%m-%d")

                return dt_obj.isoformat()

            # 【修改点】只捕获值错误和类型错误，不吞掉其他系统错误
            except (ValueError, TypeError):
                print(f"警告: 日期字段 '{col_name}' 值 '{str_value}' 无法解析，保留原值。")
                return str_value
    else:
        return str_value


# 你的 describe_features_from_dict 函数，假设它存在
def describe_features_from_dict(features_dict: dict, mapping: dict) -> list[str]:
    descriptions = []
    for key, desc_template in mapping.items():
        value = features_dict.get(key)
        if value is not None and value != '':
            if isinstance(value, bool):
                descriptions.append(desc_template.format("是" if value else "否"))
            else:
                descriptions.append(desc_template.format(value))
    return descriptions


# 你的 CABIN_FEATURES_MAP, FOOD_FEATURES_MAP, ONBOARD_FEATURES_MAP, DIVING_FEATURES_MAP
# 这些映射字典需要在这个脚本中定义
CABIN_FEATURES_MAP = {"num": "拥有{}间客舱", "aircon": "客舱{}空调", "hotwater": "客舱{}热水"}
FOOD_FEATURES_MAP = {"fullboard": "提供{}餐饮服务", "vegetarian": "{}素食选项"}
ONBOARD_FEATURES_MAP = {"tv": "船上{}电视", "lounge": "船上{}休息室"}
DIVING_FEATURES_MAP = {"nitrox": "是否提供高氧潜水:{}", "tech": "技术潜水支持:{}"}  # 示例


def process_raw_data_to_chunks(input_raw_json_file, output_chunks_json_file):
    if not os.path.exists(input_raw_json_file):
        print(f"错误: 原始数据文件 '{input_raw_json_file}' 不存在。请先运行爬取脚本。")
        return

    with open(input_raw_json_file, 'r', encoding='utf-8') as f:
        raw_data_list = json.load(f)

    all_chunks = []
    print(f"开始处理 {len(raw_data_list)} 条船宿的原始数据，进行分块和元数据标注...")

    for boat_data in raw_data_list:
        # --- 原始数据获取 ---
        boat_id_raw = boat_data.get('boatId')
        boat_name_en_raw = boat_data.get('nameEN')
        boat_name_cn_raw = boat_data.get('nameCN', '（无中文名）')
        location_name_raw = boat_data.get('locationName')
        updated_time_raw = boat_data.get('updatedTime')

        # --- 规范化基础元数据 ---
        base_metadata_boat = {
            "category": normalize_single_metadata_value("category", "船宿船舶信息"),  # 直接将字符串也走一遍规范化
            "boatId": normalize_single_metadata_value("boatId", boat_id_raw),
            "nameCN": normalize_single_metadata_value("nameCN", boat_name_cn_raw),
            "nameEN": normalize_single_metadata_value("nameEN", boat_name_en_raw),
            "locationName": normalize_single_metadata_value("locationName", location_name_raw),
            "nitrox": normalize_single_metadata_value("nitrox", boat_data.get('nitrox')),
            "wifi": normalize_single_metadata_value("wifi", boat_data.get('wifi')),
            "diving_equipment": normalize_single_metadata_value("diving_equipment", boat_data.get('diving_equipment')),
            "tech_diving_friendly": normalize_single_metadata_value("tech_diving_friendly",
                                                                    boat_data.get('tech_diving_friendly')),
            "languages": normalize_single_metadata_value("languages", boat_data.get('languages')),
            "policy": normalize_single_metadata_value("policy", "; ".join(boat_data.get('policy', []))),
            "rating": normalize_single_metadata_value("rating", boat_data.get('rating')),
            "yearBuilt": normalize_single_metadata_value("yearBuilt", boat_data.get('yearBuilt')),
            "updatedTime": normalize_single_metadata_value("updatedTime", updated_time_raw),
            "Metadata_source": normalize_single_metadata_value("Metadata_source",
                                                               f"{BASE_URL}/boat/{boat_id_raw}/detail"),
            "Metadata_file_type": normalize_single_metadata_value("Metadata_file_type", "json")
        }
        # 移除 None 值的元数据，避免存入 JSON
        base_metadata_boat = {k: v for k, v in base_metadata_boat.items() if v is not None}

        # --- Chunk 1: 船宿基本信息 (船信息) ---
        cabin_details_desc = describe_features_from_dict(boat_data.get('cabin', {}), CABIN_FEATURES_MAP)
        food_details_desc = describe_features_from_dict(boat_data.get('food', {}), FOOD_FEATURES_MAP)
        onboard_details_desc = describe_features_from_dict(boat_data.get('onboard', {}), ONBOARD_FEATURES_MAP)
        diving_details_desc = describe_features_from_dict(boat_data.get('diving', {}), DIVING_FEATURES_MAP)
        content_boat_info = (
            f"船名称: 【{base_metadata_boat.get('nameCN', '')}】({base_metadata_boat.get('nameEN', '')})。"
            f"船宿路线位于: 【{base_metadata_boat.get('locationName', '')}】。"
            f"评级: 【{base_metadata_boat.get('rating', '未知')}】分。"
            f"建成年份: 【{base_metadata_boat.get('yearBuilt', '未知')}】年。"
            f"船体材质: 【{boat_data.get('material', '未知')}】。"
            f"船长: 【{boat_data.get('length', '未知')}】，船宽: 【{boat_data.get('width', '未知')}】。"
            f"高氧供应: 【{'是' if base_metadata_boat.get('nitrox') else '否'}】。"
            f"Wi-Fi供应: 【{'是' if base_metadata_boat.get('wifi') else '否'}】。"
            f"潜水装备租赁: 【{base_metadata_boat.get('diving_equipment', '未知')}】。"
            f"技术潜水友好: 【{'是' if base_metadata_boat.get('tech_diving_friendly') else '否'}】。"
            f"支持语言: 【{base_metadata_boat.get('languages', '未知')}】。"
            f"客舱数量: 【{boat_data.get('roomNum', '未知')}】间。"
            f"船宿描述: 【{boat_data.get('descr', '暂无描述')}】。"
            f"包含费用: 【{boat_data.get('included_cn', '无')}】。"
            f"不包含费用: 【{boat_data.get('notincluded_cn', '无')}】。"
            f"船上支付方式: 【{boat_data.get('payment', '未知')}】。"
            f"政策概览: 【{base_metadata_boat.get('policy', '无')}】。"
            f"最后更新时间: 【{base_metadata_boat.get('updatedTime', '未知')}】。"
            f"客舱特点: 【{', '.join(cabin_details_desc) if cabin_details_desc else '无特殊描述'}】。"
            f"餐饮服务: 【{', '.join(food_details_desc) if food_details_desc else '无特殊描述'}】。"
            f"船上设施: 【{', '.join(onboard_details_desc) if onboard_details_desc else '无特殊描述'}】。"
            f"潜水支持设备和功能: 【{', '.join(diving_details_desc) if diving_details_desc else '无特殊描述'}】。"
        )
        all_chunks.append({
            "page_content": content_boat_info,
            "metadata": {**base_metadata_boat, "chunk_type": "船宿基本信息"}
        })

        # --- 遍历船宿下的所有行程 (schedules) ---
        processed_tours = set()
        for schedule in boat_data.get('schedules', []):
            tour_id_raw = schedule.get('tourId')
            trip_id_raw = schedule.get('tripId')

            # --- Chunk 2: 路线信息 (Tour Template Details) ---
            if tour_id_raw and tour_id_raw not in processed_tours:
                tour_details = schedule.get('tour_details', {})
                metadata_tour = {
                    "category": normalize_single_metadata_value("category", "船宿路线"),
                    "boatId": normalize_single_metadata_value("boatId", boat_id_raw),
                    "tourId": normalize_single_metadata_value("tourId", tour_id_raw),  # 确保这里被转为字符串
                    "nameCN": normalize_single_metadata_value("nameCN", schedule.get('nameCN')),
                    "nameEN": normalize_single_metadata_value("nameEN", schedule.get('nameEN')),
                    "locationName": normalize_single_metadata_value("locationName", location_name_raw),
                    "experience": normalize_single_metadata_value("experience", tour_details.get('experience')),
                    "certification": normalize_single_metadata_value("certification",
                                                                     tour_details.get('certification')),
                    "dives": normalize_single_metadata_value("dives", tour_details.get('dives')),
                    "duration": normalize_single_metadata_value("duration", tour_details.get('duration')),
                    "nights": normalize_single_metadata_value("nights", tour_details.get('nights')),
                    "updatedTime": normalize_single_metadata_value("updatedTime", updated_time_raw),
                    "Metadata_source": normalize_single_metadata_value("Metadata_source",
                                                                       f"{BASE_URL}/boat/{boat_id_raw}/detail"),
                    "Metadata_file_type": normalize_single_metadata_value("Metadata_file_type", "json")
                }
                metadata_tour = {k: v for k, v in metadata_tour.items() if v is not None}  # 移除None

                content_tour_info = (
                    f"路线ID: 【{metadata_tour.get('tourId', '未知')}】。"
                    f"路线名称: 【{metadata_tour.get('nameCN', '')}】({metadata_tour.get('nameEN', '')})。"
                    f"所属船只ID: 【{metadata_tour.get('boatId', '未知')}】。"
                    f"出发地点: 【{tour_details.get('departureLocation', '未知')}】，返程抵达地点: 【{tour_details.get('arrivalLocation', '未知')}】。"
                    f"出发港口: 【{tour_details.get('departurePort', '未知')}】，返程抵达港口: 【{tour_details.get('arrivalPort', '未知')}】。"
                    f"经验要求: 【{metadata_tour.get('experience', '无')}】。"
                    f"认证要求: 【{metadata_tour.get('certification', '无')}】。"
                    f"潜水次数: 【{metadata_tour.get('dives', '未知')}】次。"
                    f"行程天数: 【{metadata_tour.get('duration', '未知')}】天，晚数: 【{metadata_tour.get('nights', '未知')}】晚。"
                    f"包含费用: 【{tour_details.get('included_cn', '无')}】。"
                    f"不包含费用: 【{tour_details.get('notincluded_cn', '无')}】。"
                    f"行程介绍: 【{tour_details.get('programm', '无')}】。"
                )
                all_chunks.append({
                    "page_content": content_tour_info,
                    "metadata": {**metadata_tour, "chunk_type": "船宿路线"}
                })
                processed_tours.add(tour_id_raw)  # 标记此tourId已处理 (注意这里是原始tourId_raw)

            # --- Chunk 3: 旅程信息 (Trip Schedule Details) ---
            metadata_trip = {
                "category": normalize_single_metadata_value("category", "船宿行程"),
                "boatId": normalize_single_metadata_value("boatId", boat_id_raw),
                "tourId": normalize_single_metadata_value("tourId", tour_id_raw),  # 确保这里被转为字符串
                "tripId": normalize_single_metadata_value("tripId", trip_id_raw),  # 确保这里被转为字符串
                "tour_nameCN": normalize_single_metadata_value("nameCN", schedule.get('nameCN')),
                "tour_nameEN": normalize_single_metadata_value("nameEN", schedule.get('nameEN')),
                "boat_nameCN": normalize_single_metadata_value("nameCN", boat_name_cn_raw),
                "boat_nameEN": normalize_single_metadata_value("nameEN", boat_name_en_raw),
                "locationName": normalize_single_metadata_value("locationName", location_name_raw),
                "departureLocation": normalize_single_metadata_value("departureLocation", tour_details.get('departureLocation')),
                "arrivalLocation": normalize_single_metadata_value("arrivalLocation",
                                                                     tour_details.get('arrivalLocation')),
                "arrivalDate": normalize_single_metadata_value("arrivalDate", schedule.get('arrivalDate')),
                "arrivalDate_display": normalize_single_metadata_value("arrivalDate_display", schedule.get('arrivalDate_display')),
                "departureDate": normalize_single_metadata_value("departureDate", schedule.get('departureDate')),
                "departureDate_display": normalize_single_metadata_value("departureDate_display", schedule.get('departureDate_display')),
                "updatedTime": normalize_single_metadata_value("updatedTime", updated_time_raw),
                # 其他行程特有的布尔值、数字等也需要规范化
                "nitrox": normalize_single_metadata_value("nitrox", boat_data.get('nitrox')),
                "wifi": normalize_single_metadata_value("wifi", boat_data.get('wifi')),
                "tech_diving_friendly": normalize_single_metadata_value("tech_diving_friendly",
                                                                        boat_data.get('tech_diving_friendly')),
                "experience": normalize_single_metadata_value("experience", tour_details.get('experience')),
                "certification": normalize_single_metadata_value("certification", tour_details.get('certification')),
                "dives": normalize_single_metadata_value("dives", tour_details.get('dives')),
                "duration": normalize_single_metadata_value("duration", tour_details.get('duration')),
                "nights": normalize_single_metadata_value("nights", tour_details.get('nights')),
                "Metadata_source": normalize_single_metadata_value("Metadata_source",
                                                                   f"{BASE_URL}/boat/{boat_id_raw}/detail"),
                "Metadata_file_type": normalize_single_metadata_value("Metadata_file_type", "json")
            }

            metadata_trip = {k: v for k, v in metadata_trip.items() if v is not None}  # 移除None

            content_trip_info = (
                f"旅程ID: 【{metadata_trip.get('tripId', '未知')}】。"
                f"所属路线ID: 【{metadata_trip.get('tourId', '未知')}】。"
                f"所属船只ID: 【{metadata_trip.get('boatId', '未知')}】。"
                f"旅程名称: 【{metadata_trip.get('tour_nameCN', '')}】({metadata_trip.get('tour_nameEN', '')})。"
                f"出发日期: 【{metadata_trip.get('departureDate_display', '未知')}】，抵达日期: 【{metadata_trip.get('arrivalDate_display', '未知')}】。"
                f"高氧供应: 【{'是' if metadata_trip.get('nitrox') else '否'}】。"
                f"Wi-Fi供应: 【{'是' if metadata_trip.get('wifi') else '否'}】。"
                f"技术潜水友好: 【{'是' if metadata_trip.get('tech_diving_friendly') else '否'}】。"
                f"潜水次数: 【{metadata_trip.get('dives', '未知')}】次。"
                f"行程天数: 【{metadata_trip.get('duration', '未知')}】天，晚数: 【{metadata_trip.get('nights', '未知')}】晚。"
                f"最后更新时间: 【{metadata_trip.get('updatedTime', '未知')}】。"
                f"该旅程的预订政策: 【{schedule.get('booking_policy', '无')}】。"
                f"该旅程的取消政策: 【{schedule.get('cancellation_policy', '无')}】。"
                f"实时空位(非实时): 【{schedule.get('availableCount_raw', '未知')}】。"
                f"实时价格(非实时): 【{schedule.get('price_raw', {}).get('value', '未知')}{schedule.get('price_raw', {}).get('unit', '')}】。"
            )
            all_chunks.append({
                "page_content": content_trip_info,
                "metadata": {**metadata_trip, "chunk_type": "船宿行程"}
            })

    # 保存分块后的数据到JSON文件
    try:
        with open(output_chunks_json_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=4)
        print(f"\n所有船宿详情数据（分块后）已成功保存到 '{output_chunks_json_file}'")
        print(f"共生成了 {len(all_chunks)} 个文本块。")
    except Exception as e:
        print(f"保存数据到文件 '{output_chunks_json_file}' 失败: {e}")


if __name__ == "__main__":
    import os

    # 清理crawled_boat_ids.json (仅测试用，实际生产环境不要轻易删除)
    # if os.path.exists(CRAWLED_IDS_FILE):
    #     os.remove(CRAWLED_IDS_FILE)
    #     print(f"已删除旧的 {CRAWLED_IDS_FILE} 文件，将从头开始爬取。")
    raw_output_file = "cooldive_boat_details_with_trips_1-10.json"

    # print("\n--- 示例1: 增量爬取列表页中前5个“新”船宿的详情和每个船宿最多2条行程 ---")
    # get_cooldive_liveaboard_detail(
    #     output_json_file=raw_output_file,
    #     max_list_pages=1,  # 每次扫描列表页的第一页来寻找新ID
    #     batch_size=10,  # 目标获取X个新船宿的详情
    #     max_trips_per_boat=5  # 每个船宿最多爬取X条行程
    # )


    # --- 阶段2: 处理原始数据，分块并标注元数据 ---
    print("\n======== 阶段2: 分块并标注元数据 ========")
    if os.path.exists(raw_output_file):
        process_raw_data_to_chunks(
            input_raw_json_file=raw_output_file,
            output_chunks_json_file="docs/cooldive_liveaboard_chunks_data_1-10.json"
        )
    else:
        print(f"原始数据文件 '{raw_output_file}' 不存在，跳过分块处理。")

    print("\n脚本执行完毕。")