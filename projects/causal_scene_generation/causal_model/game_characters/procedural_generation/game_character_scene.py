from PIL import ImageOps, Image
import os

image_dict = {
        "Satyr": {
            "base_path": "../images/satyr/PNG/",
            "Attacking": "/reference/Attacking/attack.png",
            "Taunt": "/reference/Taunt/taunt.png",
            "Walking": "/reference/Walking/walking.png",
            "Dying": "/reference/Dying/dying.png",
            "Hurt": "/reference/Hurt/hurt.png",
            "Idle": "/reference/Idle/idle.png"
        },
        "Golem": {
            "base_path": "../images/golem/PNG/",
            "Attacking": "/reference/Attacking/attack.png",
            "Taunt": "/reference/Taunt/taunt.png",
            "Walking": "/reference/Walking/walking.png",
            "Dying": "/reference/Dying/dying.png",
            "Hurt": "/reference/Hurt/hurt.png",
            "Idle": "/reference/Idle/idle.png"
        }
    }
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def draw_duel(actor, reactor):
    '''
    Loading variables.
    '''
    act_name = actor["name"]
    rct_name = reactor["name"]
    action = actor["action"]
    reaction = reactor["reaction"]
    act_type = actor["type"]
    rct_type = reactor["type"]

    
    img1 = Image.open(image_dict[act_name]["base_path"]+act_type+image_dict[act_name][action])
    img2 = Image.open(image_dict[rct_name]["base_path"]+rct_type+image_dict[rct_name][reaction])
    
    #Flipping the reactor to give the feel of a duel.
    img2 = ImageOps.mirror(img2)
    
    return get_concat_h(img1, img2), img1, img2