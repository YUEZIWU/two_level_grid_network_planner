'''
THIS SCRIPT DOES THE FOLLOWING:
    1) Picks a specific ward and gets all the structures in ward
    2) Saves them to pickle

*** yuezi updated for uganda network planning March 08, 2023
'''



# import pickle
# import fiona
# from fiona.crs import from_epsg
# import simone_agg_clustering as ac
# from rtree import index
import re
import os
import sys
import pandas as pd
from shapely.geometry import Point
import numpy as np
import geopandas as gpd


def cleanNames(name):
    name = name.title()  # converts to CamelCase
    valids = re.findall(r"[\w']+", name)  # Removes all unwanted xters
    valids = ''.join(valids)
    return (valids)


def createIndex(structures):
    # initialize index
    idx = index.Index()
    for pos, row in structures.iterrows():
        try:
            left, bottom, right, top = pointBBox(row)
            idx.insert(int(pos), (left, bottom, right, top))
        except:
            print(row)
    return (idx)


def pointBBox(row):
    x, y = row['gps_x'], row['gps_y']
    return (x, y, x, y)


def jointBBox(poly):
    left, bottom, right, top = [], [], [], []
    for geom in poly.geoms:
        left.append(geom.bounds[0])
        bottom.append(geom.bounds[1])
        right.append(geom.bounds[2])
        top.append(geom.bounds[3])
    return ((min(left), min(bottom), max(right), max(top)))


def structuresInPoly(poly, idx):
    largestBB = jointBBox(poly)
    structureIdx_in_polygon = list(idx.intersection(largestBB))  ### returns index of structures in each polygon
    return (structureIdx_in_polygon)


def getPointsInLevel(structures, poly):
    df = pd.DataFrame(columns=['utm_x', 'utm_y', 'gps_x', 'gps_y'])
    for ind, row in structures.iterrows():
        bldgPoint = Point(row[['gps_x', 'gps_y']])
        for geom in poly.geoms:
            if geom.contains(bldgPoint):
                df = df.append(
                    {'utm_x': row['utm_x'], 'utm_y': row['utm_y'], 'gps_x': row['gps_x'], 'gps_y': row['gps_y']},
                    ignore_index=True)
    return (df)


def getSaveToken(levelDir, ward, county):
    return (levelDir + '/' + ward + '_' + county + '.pck')


def loggers(log_filename, data):
    with open(log_filename, 'w') as hdl:
        pickle.dump(data, hdl, protocol=pickle.HIGHEST_PROTOCOL)


def getAllStructs(structureIdx_in_polygon, structures, poly, ward, county, levelDir):
    IDXlist = [int(k) for k in structureIdx_in_polygon]  ## verify on cluster might not need list version
    try:
        curDF = structures.loc[IDXlist]
        ## checkif DF is not empty
        if not curDF.empty:
            inShapeStructs = getPointsInLevel(curDF, poly)
            print('Number of structures in ', ward, county, ':', inShapeStructs.shape[0])
            savetoken = getSaveToken(levelDir, ward, county)
            loggers(savetoken, inShapeStructs)
            print('Saved structures in ', ward)
    except Exception as LErr:
        print('Looping Error for index:', LErr)
    return inShapeStructs


def is_ward_merged(ward, county, data):
    valid = False
    for f in data:
        if ward in f and county in f:
            valid = True
    return valid


# def make_save_token(ward,county):
#     token = ward + '_' + county
#     outputDir = '../../all_wards_20m/'+token
#     if not os.path.exists(outputDir):
#         os.mkdir(outputDir)
#     savepath = outputDir+ '/'+token + '.shp'
#     return savepath


def make_savetoken(srcpath, idx):
    subgrid = 'subgrid' + '_' + str(idx)

    if not os.path.exists(os.path.join(srcpath, subgrid)):
        os.mkdir(os.path.join(srcpath, subgrid))

    savetoken = os.path.join(srcpath, subgrid) + '/' + subgrid + '.shp'
    return savetoken


def make_save_dir(district):
    if not os.path.exists('../../data/district_split_results'):
        os.mkdir('../../data/district_split_results')
    token = district
    outputDir = '../../data/district_split_results/' + token
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)
    savepath = outputDir + '/'
    return savepath


def make_shp(savepath, structs):
    geometry = [Point(xy) for xy in zip(structs.utm_x, structs.utm_y)]
    structs_geo = gpd.GeoDataFrame(structs, geometry=geometry)
    structs_geo = structs_geo.set_crs('epsg:32636')
    structs_geo = structs_geo[['geometry']]
    structs_geo.to_file(savepath, driver='ESRI Shapefile')


def get_extent(structures):
    '''
    Get the structure extent and add a 10 m buffer
    :param structures:
    :return:
    '''
    minx = min(structures['utm_x']) - 10
    maxx = max(structures['utm_x']) + 10
    miny = min(structures['utm_y']) - 10
    maxy = max(structures['utm_y']) + 10
    return minx, miny, maxx, maxy


def make_df(src, bbox):
    df = []
    count = 0
    for idx, elem in src.iterrows():
        if int(np.floor(elem['utm_x'])) >= bbox[0] and int(np.floor(elem['utm_x'] < bbox[2])):
            if int(np.floor(elem['utm_y'])) >= bbox[1] and int(np.floor(elem['utm_y'] < bbox[-1])):
                df.append([elem['utm_x'], elem['utm_y']])
                count += 1
    df = pd.DataFrame(df, columns=['utm_x', 'utm_y'])
    return df


def recursive_split(start_bbox, src,numStructures,clusterSize,radius):
    stack, valid = [start_bbox], []
    while stack:
        bbox = stack.pop()
        cur_df = make_df(src, bbox)
        if cur_df.shape[0] < numStructures:
            if cur_df.shape[0] <= clusterSize:
                valid.append(bbox)
            else:
                y_step = (bbox[-1] - bbox[1]) / 2.0
                x_step = (bbox[2] - bbox[0]) / 2.0
                cur_radius = int(((y_step ** 2) + (x_step ** 2)) ** 0.5)
                if cur_radius <= radius:
                    valid.append(bbox)
                else:
                    for x in np.arange(bbox[0], bbox[2], x_step):
                        for y in np.arange(bbox[1], bbox[-1], y_step):
                            sub_bbox = x, y, x + x_step, y + y_step
                            sub_bbox = [int(np.floor(b)) for b in sub_bbox]
                            stack.append(sub_bbox)
        else:
            y_step = (bbox[-1] - bbox[1]) / 2.0
            x_step = (bbox[2] - bbox[0]) / 2.0
            cur_radius = int(((y_step ** 2) + (x_step ** 2)) ** 0.5)
            if cur_radius <= radius:
                valid.append(bbox)
            else:
                for x in np.arange(bbox[0], bbox[2], x_step):
                    for y in np.arange(bbox[1], bbox[-1], y_step):
                        sub_bbox = x, y, x + x_step, y + y_step
                        sub_bbox = [int(np.floor(b)) for b in sub_bbox]
                        stack.append(sub_bbox)
    return valid


def split_by_max_nodes(start_bbox, structures):
    stack, valid = [start_bbox], []
    while stack:
        bbox = stack.pop()
        print(bbox)
        cur_df = make_df(structures, bbox)
        max_nodes = ac.get_max_nodes_in_cluster(cur_df)
        print('Max Nodes: ', max_nodes)
        if max_nodes < 300:
            valid.append(bbox)
        else:
            y_step = (bbox[-1] - bbox[1]) / 2.0
            x_step = (bbox[2] - bbox[0]) / 2.0
            if x_step < 500 or y_step < 500:
                valid.append(bbox)
            else:
                for x in np.arange(bbox[0], bbox[2], x_step):
                    for y in np.arange(bbox[1], bbox[-1], y_step):
                        sub_bbox = x, y, x + x_step, y + y_step
                        sub_bbox = [int(np.floor(b)) for b in sub_bbox]
                        stack.append(sub_bbox)
        print('Stack size: ', len(stack))
    return valid


def save_splits(splits, structures, ward_dir):
    print('Number of Acceptable extents:', len(splits))
    rand_ext = np.random.randint(low=0, high=len(splits))
    # print('Sample acceptable extent:', splits[rand_ext])
    print('Starting the make shapefile process')
    idx = 0
    for pos, split in enumerate(splits):
        cur_df = make_df(structures, split)
        if not cur_df.empty:
            savetoken = make_savetoken(ward_dir, idx)
            make_shp(savetoken, cur_df)
            idx += 1


def geojson_to_csv_table(geo_file):
    output = geo_file
    output.index=range(len(geo_file))
    output = output.assign(utm_x=output.loc[:, 'geometry'].x)
    output = output.assign(utm_y=output.loc[:, 'geometry'].y)
    output = output.drop(columns='geometry')
    return output


def main(merged_structures_folder,numStructures,clusterSize,radius):
    merged_geojson_files = os.listdir(merged_structures_folder)
    for merged_geojson in merged_geojson_files:
        district = merged_geojson.split('_')[0]
        print(f'{district} splitting running')
        all_structs_geo = gpd.read_file(os.path.join(merged_structures_folder, merged_geojson))
        all_structs = geojson_to_csv_table(all_structs_geo)

        if not all_structs.empty:
            try:
                save_dir = make_save_dir(district)
                bbox = get_extent(all_structs)
                bbox = [int(np.floor(b)) for b in bbox]
                valid_splits = recursive_split(bbox, all_structs,numStructures,clusterSize,radius)
                save_splits(valid_splits, all_structs, save_dir)
            except Exception as err:
                print(err)
        else:
            print('No structures found for ', district)


if __name__ == '__main__':
    merged_structures_folder = '../../data/merged_structures_geojson'
    numStructures = 3000
    clusterSize = 1000 #300
    radius = 700
    main(merged_structures_folder,numStructures,clusterSize,radius)
