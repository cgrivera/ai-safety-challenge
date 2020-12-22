# ©2020 Johns Hopkins University Applied Physics Laboratory LLC.
import cv2
import numpy as np
import math, random, time

IMG_SZ = 128         #how big is the output image
UNITY_SZ = 100.0     #what is the side length of the space in unity coordintaes
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

def draw_bullet(image, x, y):
    cv2.circle(image, (int(x),int(y)), 2, 255.0, thickness=1)
    return image

def draw_tanks_in_channel(tank_data, reference_tank):

    img = np.zeros((IMG_SZ, IMG_SZ, 1), np.uint8)

    #draw tanks
    for td in tank_data:

        if td[3] <= 0.0:
            continue

        rel_x, rel_y = point_relative_point_heading([td[0],td[1]], reference_tank[0:2], reference_tank[2])

        x = (rel_x/UNITY_SZ) * SCALE + float(IMG_SZ)*0.5
        y = (rel_y/UNITY_SZ) * SCALE + float(IMG_SZ)*0.5
        heading = td[2]
        health = td[3]

        rel_heading = heading - reference_tank[2]

        img = draw_arrow(img, x, y, rel_heading, health)

        # draw bullet if present
        if len(td) > 4:
            bx = td[4]
            by = td[5]

            if bx < 900:
                rel_x, rel_y = point_relative_point_heading([bx, by], reference_tank[0:2], reference_tank[2])
                x = (rel_x/UNITY_SZ) * SCALE + float(IMG_SZ)*0.5
                y = (rel_y/UNITY_SZ) * SCALE + float(IMG_SZ)*0.5
                img = draw_bullet(img, x, y)

    return img

def barriers_for_player(barriers, reference_tank):

    img = np.zeros((IMG_SZ, IMG_SZ, 1), np.uint8)

    # constants
    wall_value = 255
    border_allowance_global = -1.0

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
    threshold_indices = res > 0.05
    res[threshold_indices] = 1.0
    res *= 255.0

    #draw in larger image
    scl = int(SCALE)
    barrier_img = np.ones((scl*2, scl*2), np.uint8)
    barrier_img[(scl//2):3*scl//2, (scl//2):3*scl//2] = res

    #translate
    dx = (reference_tank[0]/UNITY_SZ) * SCALE
    dy = (reference_tank[1]/UNITY_SZ) * SCALE
    M = np.float32([[1,0,-dx],[0,1,-dy]])
    barrier_img = cv2.warpAffine(barrier_img,M,(scl*2,scl*2))

    #rotate about center
    ang = reference_tank[2]*(180.0/3.14)
    M = cv2.getRotationMatrix2D((float(scl*2)*0.5,float(scl*2)*0.5),ang,1)
    barrier_img = cv2.warpAffine(barrier_img,M,(scl*2,scl*2))

    #extract central area without padding
    padd = (IMG_SZ-int(SCALE))//2
    barrier_img = barrier_img[(scl//2)-padd:(scl//2)+IMG_SZ-padd, (scl//2)-padd:(scl//2)+IMG_SZ-padd]

    #add channel
    barrier_img = np.expand_dims(barrier_img, axis=2)

    #concat the walls and the barriers
    ch = np.maximum(img, barrier_img)

    return ch
# expects state data chopped on a tank by tank basis
# ie. for 5 red, 5 blue, 2 neutral, expects a length 12 array
def minimap_for_player(tank_data_original, tank_idx, barriers):

    barriers = np.flipud(barriers)

    tank_data = []
    for td in tank_data_original:
        tank_data.append([td[0], -td[1], td[2], td[3], td[4], -td[5]])

    my_data = tank_data[tank_idx]

    if my_data[3] <= 0.0:
        #display_cvimage("tank"+str(tank_idx), np.zeros((IMG_SZ, IMG_SZ, 3)))
        return np.zeros((IMG_SZ,IMG_SZ,4), dtype=np.float32)

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

    # image = np.asarray([ally_channel, enemy_channel, barriers_channel]).astype(np.float32)
    # image = np.squeeze(image)

    #print(image.shape)
    #display_cvimage("tank"+str(tank_idx), np.transpose(image,(1,2,0)))

    ret = np.asarray([ally_channel, neutral_channel, enemy_channel, barriers_channel]).astype(np.float32) / 255.0

    ret = np.squeeze(np.array(ret).transpose((3,1,2,0)))
    return ret

def display_cvimage(window_name, img):

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name,  img)
    cv2.waitKey(1)

    #time.sleep(0.2)


def displayable_rgb_map(minimap):
    ally = minimap[:,:,0]*255.0
    neutral = minimap[:,:,1]*255.0
    enemy = minimap[:,:,2]*255.0
    barrier = minimap[:,:,3]*255.0

    r_ch = np.maximum(ally, barrier)
    g_ch = np.maximum(neutral, barrier)
    b_ch = np.maximum(enemy, barrier)

    ret = np.asarray([b_ch, g_ch, r_ch]).astype(np.uint8) #cv2 actually uses bgr
    return ret.transpose((1,2,0))


if __name__ == "__main__":

    # 12 random tanks
    tank_data = []
    for i in range(12):
        x = 500
        y = 500
        h = 0
        hp = 100.0
        tank_data.append([x, y, h, hp])

    #ally
    tank_data[0] = [-15,15,0,100]
    tank_data[1] = [15,20,0,100]
    tank_data[5] = [-15,-15,3.14,100]
    tank_data[6] = [15,-20,3.14,100]

    no_barriers = np.zeros((40,40,1))


    for i in range(1000):

        tank_data[0][2] += 0.1
        # tank_data[1][2] -= 0.1

        # get image for tank 2
        minimap0 = minimap_for_player(tank_data, 0, no_barriers)
        minimap1 = minimap_for_player(tank_data, 1, no_barriers)
        minimap6 = minimap_for_player(tank_data, 5, no_barriers)
        minimap6 = minimap_for_player(tank_data, 6, no_barriers)

        #display the channels
        # display_cvimage("allies2", minimap0[1])
        # display_cvimage("allies3", minimap1[1])

        #pause for a few seconds so we can view the images
        time.sleep(0.2)
