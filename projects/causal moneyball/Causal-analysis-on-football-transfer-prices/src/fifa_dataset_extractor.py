import re
import requests
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup


if __name__ == "__main__":
    base_url = "https://www.fifaindex.com/players/"
    year_list = range(8, 21)
    fifa_df = pd.DataFrame(columns={"year", "player_name", "ovr", "pot", "nationality"})
    for year in tqdm(year_list):
        fifa_year = "fifa" + "{}".format(year).zfill(2) + "/"
        page_num = 1
        while True:
            response = requests.get(base_url+fifa_year+str(page_num))
            if response.ok:
                soup = BeautifulSoup(response.text, "lxml")
                table_rows = soup.find_all(name="tr",
                                           attrs={"data-playerid": re.compile("\d+")})
                for row in table_rows:
                    # player name
                    player = row.find(name="a",
                                           attrs={"class":"link-player"})["title"]
                    player = re.sub(pattern=re.compile("fifa(\s*)\d+", flags=re.IGNORECASE), string=player, repl="")
                    
                    ratings = row.findAll(name="span",
                                          attrs={"class": re.compile("badge(.*)rating(.*)")})
                    # overall rating
                    ovr = ratings[0].text
                    if ovr.isdigit():
                        ovr = int(ovr)
                    else:
                        ovr = None
        
                    # potential rating
                    pot = ratings[1].text
                    if pot.isdigit():
                        pot = int(pot)
                    else:
                        pot = None
        
                    # nationality
                    nationality = row.find(name="a",
                                           attrs={"class":"link-nation"})["title"]
        
                    fifa_df = fifa_df.append({"year": year,
                                              "ovr": ovr,
                                              "pot": pot,
                                              "nationality": nationality,
                                              "player_name": player}, ignore_index=True)
                page_num += 1
            else:
                break
    fifa_df.to_csv("data/fifa_stats.csv", index=False)
