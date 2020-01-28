
import cv2
import numpy as np
import math, random, time

IMG_SZ = 128         #how big is the output image
UNITY_SZ = 80.0     #what is the side length of the space in unity coordintaes
SCALE = 120.0        #what is the side length of the space when drawn on our image

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
    arrow_size = 2.0
    angle_dev = 2.0
    value = int((health/100.0)*128.0 + 127.0)

    #make verticies of an asteroid-like arrow shape
    h = heading + 1.57
    pnose = point_offset_point([x, y], h, 2.0*arrow_size)
    pleft = point_offset_point([x, y], h-angle_dev, arrow_size)
    porg = [x, y]
    pright = point_offset_point([x, y], h+angle_dev, arrow_size)
    verts = np.asarray([[pnose, pleft, porg, pright]], dtype=np.int32)

    #draw arrow onto the image and return
    cv2.fillPoly(image, verts, value)
    return image

def draw_tanks_in_channel(tank_data, reference_tank):

    img = np.zeros((IMG_SZ, IMG_SZ, 1), np.uint8)

    #draw tanks
    for td in tank_data:

        rel_x, rel_y = point_relative_point_heading([td[0],td[1]], reference_tank[0:2], reference_tank[2])

        x = (rel_x/UNITY_SZ) * SCALE + float(IMG_SZ)*0.5
        y = (rel_y/UNITY_SZ) * SCALE + float(IMG_SZ)*0.5
        heading = td[2]
        health = td[3]

        rel_heading = heading - reference_tank[2]

        img = draw_arrow(img, x, y, rel_heading, health)

    return img

def barriers_for_player(barriers, reference_tank):

    img = np.zeros((IMG_SZ, IMG_SZ, 1), np.uint8)

    # constants
    wall_value = 255
    border_allowance_global = 2.0

    unity32 = UNITY_SZ*1.5
    unityhf = UNITY_SZ*0.5

    #draw walls
    wleft = [[-unity32, -unity32], [-unityhf-border_allowance_global, -unity32], [-unityhf-border_allowance_global, unity32], [-unity32, unity32]] #in world
    wleft_rel = np.asarray([points_relative_point_heading(wleft, reference_tank[0:2], reference_tank[2])])
    wleft_rel = (wleft_rel/UNITY_SZ) * SCALE + float(IMG_SZ)*0.5
    cv2.fillPoly(img, wleft_rel.astype(np.int32), wall_value)

    wright = [[unityhf+border_allowance_global, -unity32], [unity32, -unity32], [unity32, unity32], [unityhf+border_allowance_global, unity32]] #in world
    wright_rel = np.asarray([points_relative_point_heading(wright, reference_tank[0:2], reference_tank[2])])
    wright_rel = (wright_rel/UNITY_SZ) * SCALE + float(IMG_SZ)*0.5
    cv2.fillPoly(img, wright_rel.astype(np.int32), wall_value)

    wtop = [[-unity32, -unity32], [unity32, -unity32], [unity32, -unityhf-border_allowance_global], [-unity32, -unityhf-border_allowance_global]] #in world
    wtop_rel = np.asarray([points_relative_point_heading(wtop, reference_tank[0:2], reference_tank[2])])
    wtop_rel = (wtop_rel/UNITY_SZ) * SCALE + float(IMG_SZ)*0.5
    cv2.fillPoly(img, wtop_rel.astype(np.int32), wall_value)

    wbot = [[-unity32, unityhf+border_allowance_global], [unity32, unityhf+border_allowance_global], [unity32, unity32], [-unity32, unity32]] #in world
    wbot_rel = np.asarray([points_relative_point_heading(wbot, reference_tank[0:2], reference_tank[2])])
    wbot_rel = (wbot_rel/UNITY_SZ) * SCALE + float(IMG_SZ)*0.5
    cv2.fillPoly(img, wbot_rel.astype(np.int32), wall_value)

    #draw internal barriers
    res = cv2.resize(np.squeeze(barriers[:,:,0]), dsize=(int(SCALE),int(SCALE)), interpolation=cv2.INTER_CUBIC)
    #res = np.expand_dims(res,axis=2)

    barrier_img = np.zeros((IMG_SZ, IMG_SZ), np.uint8)
    scale_int_padd = IMG_SZ-int(SCALE)
    barrier_img[(scale_int_padd//2):IMG_SZ-(scale_int_padd//2), (scale_int_padd//2):IMG_SZ-(scale_int_padd//2)] = res
    barrier_img = np.flipud(barrier_img)

    #translate
    dx = (reference_tank[0]/UNITY_SZ) * SCALE
    dy = (reference_tank[1]/UNITY_SZ) * SCALE
    M = np.float32([[1,0,-dx],[0,1,-dy]])
    barrier_img = cv2.warpAffine(barrier_img,M,(IMG_SZ,IMG_SZ))

    #rotate about center
    ang = reference_tank[2]*(180.0/3.14)
    M = cv2.getRotationMatrix2D((float(IMG_SZ)*0.5,float(IMG_SZ)*0.5),ang,1)
    barrier_img = cv2.warpAffine(barrier_img,M,(IMG_SZ,IMG_SZ))

    #add channel
    barrier_img = np.expand_dims(barrier_img, axis=2)

    #concat the walls and the barriers
    ch = np.maximum(img, barrier_img)

    return ch
# expects state data chopped on a tank by tank basis
# ie. for 5 red, 5 blue, 2 neutral, expects a length 12 array
def minimap_for_player(tank_data, tank_idx, barriers):

    data = np.asarray(tank_data)

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
    #barriers_channel = np.zeros((84, 84, 1), np.uint8)
    #barriers_channel[:80,:80,0] = barriers[:,:,0]
    barriers_channel = barriers_for_player(barriers, my_data)
    #flip images for red team so they are on the correct side of the map from their POV
    # if flip:
    #     this_channel = np.fliplr(np.flipud(this_channel))
    #     ally_channel = np.fliplr(np.flipud(ally_channel))
    #     enemy_channel = np.fliplr(np.flipud(enemy_channel))
    #     neutral_channel = np.fliplr(np.flipud(neutral_channel))

    image = np.asarray([ally_channel, enemy_channel, barriers_channel]).astype(np.float32)
    image = np.squeeze(image)
    #print(image.shape)
    display_cvimage("tank"+str(tank_idx), np.transpose(image,(1,2,0)))

    return np.asarray([ally_channel, enemy_channel, neutral_channel, barriers_channel]).astype(np.float32) / 255.0


def display_cvimage(window_name, img):

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name,  img)
    cv2.waitKey(1)

    time.sleep(2)



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
