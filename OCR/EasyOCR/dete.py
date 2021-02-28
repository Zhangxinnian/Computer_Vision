import PIL
from PIL import ImageDraw , ImageFont
im = PIL.Image.open("./examples/meet.PNG")
import easyocr
reader = easyocr.Reader(['ug'])
bounds = reader.readtext('./examples/meet.PNG')
print(bounds)

def draw_boxes(image, bounds, color='yellow', width=2):
    font = ImageFont.truetype('./STXINWEI.TTF', 6)
    draw = ImageDraw.Draw(image)
    for bound in bounds:
        p0, p1, p2, p3 = bound[0]
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
        text = str(bound[1])
        print(text)
        draw.text((p0[0],p0[1]), text, (255,0,0), font=font)
    return image

draw_boxes(im, bounds)
im.show()




