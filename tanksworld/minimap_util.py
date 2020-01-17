
import cv2
import numpy as np
import math, random, time

def point_offset_point(p_origin, angle, radius):
    px = p_origin[0] + math.cos(angle)*radius
    py = p_origin[1] + math.sin(angle)*radius
    return px, py

def point_relative_point_heading(point, new_origin, heading):
    
    #get point (x,y) as (radius, angle)
    dx = point[0] - new_origin[0]
    dy = point[1] - new_origin[1]
    angle = math.atan2(dy, dx)
    rad = math.sqrt(dx*dx + dy*dy)

    #rotate
    angle -= heading

    #recover x and y relative to point
    nx = rad * math.cos(angle)
    ny = rad * math.sin(angle)

    return [nx, ny]

def points_relative_point_heading(points, new_origin, heading):
    return [point_relative_point_heading(p, new_origin, heading) for p in points]


def draw_arrow(image, x, y, heading, health):

    #value of the tank is how much hp it has
    arrow_size = 3.0
    angle_dev = 2.0
    value = int((health/100.0)*128.0 + 127.0)

    #make verticies of an asteroid-like arrow shape
    h = heading - 1.57
    pnose = point_offset_point([x, y], h, 2.0*arrow_size)
    pleft = point_offset_point([x, y], h-angle_dev, arrow_size)
    porg = [x, y]
    pright = point_offset_point([x, y], h+angle_dev, arrow_size)
    verts = np.asarray([[pnose, pleft, porg, pright]], dtype=np.int32)

    #draw arrow onto the image and return
    cv2.fillPoly(image, verts, value)
    return image

def draw_tanks_in_channel(tank_data, reference_tank):

    img = np.zeros((84, 84, 1), np.uint8)


    # constants
    scale = 60.0
    wall_value = 60
    border_allowance_global = 2.0

    #draw walls
    wleft = [[-120.0, -120.0], [-40.0-border_allowance_global, -120.0], [-40.0-border_allowance_global, 120.0], [-120.0, 120.0]] #in world
    wleft_rel = np.asarray([points_relative_point_heading(wleft, reference_tank[0:2], reference_tank[2])])
    wleft_rel = (wleft_rel/80.0) * scale + 42.0
    cv2.fillPoly(img, wleft_rel.astype(np.int32), wall_value)

    wright = [[40.0+border_allowance_global, -120.0], [120.0, -120.0], [120.0, 120.0], [40.0+border_allowance_global, 120.0]] #in world
    wright_rel = np.asarray([points_relative_point_heading(wright, reference_tank[0:2], reference_tank[2])])
    wright_rel = (wright_rel/80.0) * scale + 42.0
    cv2.fillPoly(img, wright_rel.astype(np.int32), wall_value)

    wtop = [[-120.0, -120.0], [120.0, -120.0], [120.0, -40.0-border_allowance_global], [-120.0, -40.0-border_allowance_global]] #in world
    wtop_rel = np.asarray([points_relative_point_heading(wtop, reference_tank[0:2], reference_tank[2])])
    wtop_rel = (wtop_rel/80.0) * scale + 42.0
    cv2.fillPoly(img, wtop_rel.astype(np.int32), wall_value)

    wbot = [[-120.0, 40.0+border_allowance_global], [120.0, 40.0+border_allowance_global], [120.0, 120.0], [-120.0, 120.0]] #in world
    wbot_rel = np.asarray([points_relative_point_heading(wbot, reference_tank[0:2], reference_tank[2])])
    wbot_rel = (wbot_rel/80.0) * scale + 42.0
    cv2.fillPoly(img, wbot_rel.astype(np.int32), wall_value)


    #draw tanks
    for td in tank_data:

        rel_x, rel_y = point_relative_point_heading([td[0],td[1]], reference_tank[0:2], reference_tank[2])

        x = (rel_x/80.0) * scale + 42.0
        y = (rel_y/80.0) * scale + 42.0
        heading = td[2]
        health = td[3]

        rel_heading = heading - reference_tank[2]

        img = draw_arrow(img, x, y, rel_heading, health)



    return img

# expects state data chopped on a tank by tank basis
# ie. for 5 red, 5 blue, 2 neutral, expects a length 12 array
def minimap_for_player(tank_data, tank_idx):

    my_data = tank_data[tank_idx]

    if tank_idx < 5:
        ally = tank_data[:5]
        enemy = tank_data[5:10]
        neutral = tank_data[10:]
        flip = True
    else:
        enemy = tank_data[:5]
        ally = tank_data[5:10]
        neutral = tank_data[10:]
        flip = False

    this_channel = draw_tanks_in_channel([my_data], my_data)
    ally_channel = draw_tanks_in_channel(ally, my_data)
    enemy_channel = draw_tanks_in_channel(enemy, my_data)
    neutral_channel = draw_tanks_in_channel(neutral, my_data)

    #flip images for red team so they are on the correct side of the map from their POV
    # if flip:
    #     this_channel = np.fliplr(np.flipud(this_channel))
    #     ally_channel = np.fliplr(np.flipud(ally_channel))
    #     enemy_channel = np.fliplr(np.flipud(enemy_channel))
    #     neutral_channel = np.fliplr(np.flipud(neutral_channel))

    return np.asarray([this_channel, ally_channel, enemy_channel, neutral_channel]).astype(np.float32) / 255.0


def display_cvimage(window_name, img):

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)    
    cv2.imshow(window_name,  img)
    cv2.waitKey(1)





if __name__ == "__main__":

    # 12 random tanks
    tank_data = []
    for i in range(12):
        x = random.uniform(-40, 40)
        y = random.uniform(-40, 40)
        h = random.uniform(-3.14, 3.14)
        hp = random.uniform(0.0, 100.0)
        tank_data.append([x, y, h, hp])


    for i in range(1000):

        tank_data[2][2] += 0.1

        # get image for tank 2
        minimap = minimap_for_player(tank_data, 2)

        #display the channels
        display_cvimage("allies2", minimap[1])

        # get image for tank 3
        minimap = minimap_for_player(tank_data, 3)

        #display the channels
        display_cvimage("allies3", minimap[1])

        #pause for a few seconds so we can view the images
        time.sleep(0.1)