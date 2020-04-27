import re
import wikipedia
import wptools
import pickle
from tqdm import tqdm

import pandas as pd


def get_player_names(path):
    data = pd.read_csv(path)
    return data.player_name.drop_duplicates().to_frame().reset_index(drop=True)


def get_wikipedia_page(player):
#    print("Player name: {}".format(player))
    flag = 1
    while flag > 0 and flag < 5:
        try:
            player = wikipedia.search("{} footballer".format(player), results=1)[0]
            flag = 0
        except ConnectionResetError as e:
            print("\n", e, "for player {}".format(player))
            flag += 1
        except Exception as e:
            print("\n", e, "for player {}".format(player))
            flag = 0
    try:
        p_page = wptools.page(player, silent=True).get_parse(show=False)
#        print(p_page.data['title'])
    except Exception as e:
        raise ValueError(e) from None
    return p_page


def extract_infobox_from_wp(players, pickle_path):
    player_stats = {}
    for player in tqdm(players.player_name):
        try:
            page = get_wikipedia_page(player)
            player_stats[player] = page.data['infobox']
        except ValueError:
            player_stats[player] = None
    with open(pickle_path, "wb") as f:
        pickle.dump(player_stats, f)


def read_players_infobox(pickle_path):
    with open(pickle_path, "rb") as f:
        player_stats = pickle.load(f)
    return player_stats


def clean_height_data(players):
    players.height.fillna(value="", inplace=True)

    # find height in m
    players.height = players["height"].apply(lambda x: re.findall("\d[.]\d+", x)[0] if re.findall("\d[.]\d+", x) else x)

    # find height in (ft + in) and cm
    height_list = []
    for height in players.height:
        if re.findall("ft(.+)in", height): # height in (ft + in)
            get_digits = re.findall("\d+", height)
            inches = int(get_digits[0])*12 + int(get_digits[1])
            meter = 0.0254 * inches
            height_list.append(str(meter))
        elif re.findall("(.+)cm", height): # height in cm
            get_digits = re.findall("\d+", height)
            cm = int(get_digits[0])
            meter = 0.01 * cm
            height_list.append(str(meter))
        else:
            height_list.append(height)

    players.height = height_list

    players.height = pd.to_numeric(players.height, errors='coerce')

    return players

def get_player_height(players, player_stats):
    height_list = []
    for player in players.player_name:
        if player_stats[player] is not None:
            if 'height' in player_stats[player].keys():
                height_list.append(player_stats[player]['height'])
                continue
        height_list.append(None)
    players['height'] = pd.Series(height_list)
    return clean_height_data(players)


def get_player_nationalities(players, player_stats):
    national_team_list = []
    for player in tqdm(players.player_name):
        if player in player_stats.keys() and player_stats[player] is not None:
            latest_national_team = 1
            while True:
                try:
                    player_stats[player]["nationalteam{}".format(latest_national_team)]
                except KeyError:
                    latest_national_team -= 1
                    break
                latest_national_team += 1
            if latest_national_team == 0:
                national_team_list.append(None)
                continue
            s = player_stats[player]["nationalteam{}".format(latest_national_team)]
            try:
                regex_s = re.search("\|(.*)\]\]", s)
                national_team_list.append(regex_s.group(1))
            except AttributeError:
                national_team_list.append(s)
        else:
            national_team_list.append(None)
    players['nationality'] = national_team_list
    players['nationality'] = players.nationality.str.replace("\w([-]|)\d{2}", "").str.strip()
    return players


def get_player_goals(players, player_stats):
    goals_df = pd.DataFrame(columns=["player_name", "from_year", 
                                     "to_year", "club", "apps", "gls"])
    for player in tqdm(players.player_name):
        if player in player_stats.keys() and player_stats[player] is not None:
            num = 1
            while True:
                try:
                    goal = player_stats[player]["goals{}".format(num)]
                    goal = int(goal) if goal.isdigit() else 0
                    
                    apps = player_stats[player]["caps{}".format(num)]
                    apps = int(apps) if apps.isdigit() else 0
                    
                    years = player_stats[player]["years{}"
                                        .format(num)].split("–")
                    
                    club = player_stats[player]["clubs{}".format(num)]
                    regex_club  = re.search("\|(.*)\]\]", club)
                    if regex_club:
                        club = regex_club.group(1)
                    else:
                        regex_club  = re.search("\[\[(.*)\]\]", club)
                        if regex_club:
                            club = regex_club.group(1)
                    
                    goals_df = goals_df.append({"player_name": player,
                                    "from_year": years[0], 
                                     "to_year": years[1] if len(years) > 1 else None,
                                     "club": club,
                                     "apps": apps,
                                     "gls": goal}, ignore_index=True)
                except KeyError:
                    num -= 1
                    break
                num += 1
            if num == 0:
                goals_df = goals_df.append({
                        "player_name": player}, ignore_index=True)
                continue
        else:
            goals_df = goals_df.append({
                        "player_name": player}, ignore_index=True)
    return goals_df


if __name__ == "__main__":
    players = get_player_names(path="data/transfers1.2.csv")
    pickle_path = "data/player_infobox_data.pkl"

    ''' Extract infoboxes from wikipedia pages of players '''
#    extract_infobox_from_wp(players, pickle_path)

    ''' Read from saved wikipedia page infoboxes '''
    player_stats = read_players_infobox(pickle_path)

    ''' Get player heights '''
    players = get_player_height(players, player_stats)

    ''' Get player nationalities '''
    players = get_player_nationalities(players, player_stats)

    ''' Get player goals '''
    goals_df = get_player_goals(players, player_stats)

    ''' Write to disk '''
    players.to_csv("data/player_wikipedia_info.csv", index=False,
                   encoding='utf-8')
    goals_df.to_csv("data/player_wikipedia_stats.csv", index=False,
                    encoding='utf-8')