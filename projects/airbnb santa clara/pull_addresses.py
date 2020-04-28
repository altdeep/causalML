import pandas as pd
import requests
import matplotlib.pyplot as plt
G_API = '#ADD GOOGLE API KEY'

# reading csv file



def filter_dataframe(df, column_name, value):
    return df.loc[df[column_name] == value]

def get_address(lat, long):
    URL = 'https://maps.googleapis.com/maps/api/geocode/json?latlng=' + str(lat) + ',' + str(long) + '&key=' + G_API
    r = requests.get(URL)
    data = r.json()
    address = data['results'][0]['formatted_address']
    return address



def generate_csv_data(listings, write_filename):
    listings.reset_index(drop=True, inplace=True)
    addresses = []
    print('##########GETTING ADDRESSES FROM GOOGLE#############')
    for index, row in listings.iterrows():
        print(row['latitude'], row['longitude'])
        address = get_address(row['latitude'], row['longitude'])
        addresses.append(address)
    listings['address'] = pd.Series(addresses)
    listings.to_csv(write_filename)
    return listings

def pull_addresses_0320(listings, write_file_name):
    filtered_listings = listings.loc[listings['property_type'].isin(['Condominium','Townhouse'])]
    filtered_listings = filter_dataframe(filtered_listings, 'room_type', 'Entire home/apt')
    filtered_listings = filtered_listings[["zipcode",
                     "bedrooms",
                     "bathrooms",
                     "price",
                     "neighbourhood_cleansed",
                     "latitude",
                     "longitude",
                    "property_type",
                    "room_type",
                    "is_location_exact"]].copy()
    generate_csv_data(filtered_listings, write_file_name)



def pull_addresses_0719(old_listings, listings, write_file_name):
    filtered_listings = old_listings.loc[old_listings['property_type'].isin(['Condominium', 'Townhouse'])]
    filtered_listings = filter_dataframe(filtered_listings, 'room_type', 'Entire home/apt')

    filtered_listings = filtered_listings[["id",
                                           "zipcode",
                                           "bedrooms",
                                           "bathrooms",
                                           "price",
                                           "neighbourhood_cleansed",
                                           "latitude",
                                           "longitude",
                                           "property_type",
                                           "room_type",
                                           "is_location_exact"]]

    filtered_listings_0719 = listings.loc[listings['property_type'].isin(['Condominium', 'Townhouse'])]
    filtered_listings_0719 = filter_dataframe(filtered_listings_0719, 'room_type', 'Entire home/apt')

    filtered_listings_0719 = filtered_listings_0719[["id",
                                                     "zipcode",
                                                     "bedrooms",
                                                     "bathrooms",
                                                     "price",
                                                     "neighbourhood_cleansed",
                                                     "latitude",
                                                     "longitude",
                                                     "property_type",
                                                     "room_type",
                                                     "is_location_exact"]]

    nonduped_rows = set(filtered_listings_0719.id) - set(filtered_listings.id)
    filtered_listings_0719 = filtered_listings_0719[filtered_listings_0719.id.isin(nonduped_rows)]

    filtered_listings_0719 = filtered_listings_0719[
        (filtered_listings_0719.bedrooms <= 3) & (filtered_listings_0719.bathrooms <= 3)]
    generate_csv_data(filtered_listings_0719, write_file_name)

listings_0320 = pd.read_csv("data/listings_0320.csv")
listings_0719 = pd.read_csv("data/listings_0719.csv")
pull_addresses_0320(listings_0320, 'data/augmented_data_0320.csv')
pull_addresses_0719(listings_0320, listings_0719, "data/augmented_data_0719.csv")