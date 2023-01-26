from PIL import Image, ImageDraw

#  print(trace[0])  # single line from textfile
#  print(trace[0].__getitem__(0))  # orientation
#  print(trace[0].__getitem__(1))  # x-coordinate
#  print(trace[0].__getitem__(2))  # y-coordinate


traces = 30


def create_image(w, h, trace, index):
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)

    # Draw initial line from starting point to first point
    start_pos = (100, 100)
    next_pos = (float(trace[0].__getitem__(1)), float(trace[0].__getitem__(2)))
    draw.line((start_pos, next_pos), fill="white", width=1)

    # Draw predator
    for t in range(len(trace)):
        if trace[t].__getitem__(0).startswith("Predator"):
            t_pos = (float(trace[t].__getitem__(1)), float(trace[t].__getitem__(2)))
            draw.line((next_pos, t_pos), fill="white", width=1)
            next_pos = t_pos

    img.save("trace" + str(index) + ".png")


for i in range(1, traces + 1):
    trace = []

    with open("files/trace" + str(i) + ".txt") as f:
        lines = [line.rstrip() for line in f]

    for line in lines:
        x = line.split(", ")
        trace.append(x)

    create_image(200, 200, trace, i)
