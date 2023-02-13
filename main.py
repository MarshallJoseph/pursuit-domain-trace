from PIL import Image, ImageDraw

#  print(trace[0])  # single line from textfile
#  print(trace[0].__getitem__(0))  # orientation
#  print(trace[0].__getitem__(1))  # x-coordinate
#  print(trace[0].__getitem__(2))  # y-coordinate

num_traces = 30
num_groups = 8


def output_images(w, h, trace, index, group):
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

    if group == 1:
        img.save("files/images/trace" + str(index) + ".png")
    elif group == 2:
        img.save("files/images/trace" + str(index) + "-xy.png")
    elif group == 3:
        img.save("files/images/trace" + str(index) + "x-y.png")
    elif group == 4:
        img.save("files/images/trace" + str(index) + "-x-y.png")
    elif group == 5:
        img.save("files/images/trace" + str(index) + "yx.png")
    elif group == 6:
        img.save("files/images/trace" + str(index) + "-yx.png")
    elif group == 7:
        img.save("files/images/trace" + str(index) + "y-x.png")
    elif group == 8:
        img.save("files/images/trace" + str(index) + "-y-x.png")


def generate_images():
    for i in range(1, num_traces + 1):
        group = 1
        for j in range(1, num_groups + 1):
            # generate trace based on group
            trace = open_trace(i, group)
            output_images(200, 200, trace, i, group)
            # reset group back to 1 if end reached
            group += 1


def group_traces():
    group = 1  # default traces
    for i in range(1, num_traces + 1):
        trace = open_trace(i, group)

        t1 = open("files/text/trace" + str(i) + "-xy.txt", "w")
        for line in trace:
            t1.write(line.__getitem__(0) + ", " + str(200 - float(line.__getitem__(1))) + ", " + str(float(line.__getitem__(2))) + "\n")

        t2 = open("files/text/trace" + str(i) + "x-y.txt", "w")
        for line in trace:
            t2.write(line.__getitem__(0) + ", " + str(float(line.__getitem__(1))) + ", " + str(200 - float(line.__getitem__(2))) + "\n")

        t3 = open("files/text/trace" + str(i) + "-x-y.txt", "w")
        for line in trace:
            t3.write(line.__getitem__(0) + ", " + str(200 - float(line.__getitem__(1))) + ", " + str(200 - float(line.__getitem__(2))) + "\n")

        t4 = open("files/text/trace" + str(i) + "yx.txt", "w")
        for line in trace:
            t4.write(line.__getitem__(0) + ", " + str(float(line.__getitem__(2))) + ", " + str(float(line.__getitem__(1))) + "\n")

        t5 = open("files/text/trace" + str(i) + "-yx.txt", "w")
        for line in trace:
            t5.write(line.__getitem__(0) + ", " + str(200 - float(line.__getitem__(2))) + ", " + str(float(line.__getitem__(1))) + "\n")

        t6 = open("files/text/trace" + str(i) + "y-x.txt", "w")
        for line in trace:
            t6.write(line.__getitem__(0) + ", " + str(float(line.__getitem__(2))) + ", " + str(200 - float(line.__getitem__(1))) + "\n")

        t7 = open("files/text/trace" + str(i) + "-y-x.txt", "w")
        for line in trace:
            t7.write(line.__getitem__(0) + ", " + str(200 - float(line.__getitem__(2))) + ", " + str(200 - float(line.__getitem__(1))) + "\n")


def open_trace(i, group):
    trace = []

    if group == 1:
        with open("files/text/trace" + str(i) + ".txt") as f:
            lines = [line.rstrip() for line in f]
    elif group == 2:
        with open("files/text/trace" + str(i) + "-xy.txt") as f:
            lines = [line.rstrip() for line in f]
    elif group == 3:
        with open("files/text/trace" + str(i) + "x-y.txt") as f:
            lines = [line.rstrip() for line in f]
    elif group == 4:
        with open("files/text/trace" + str(i) + "-x-y.txt") as f:
            lines = [line.rstrip() for line in f]
    elif group == 5:
        with open("files/text/trace" + str(i) + "yx.txt") as f:
            lines = [line.rstrip() for line in f]
    elif group == 6:
        with open("files/text/trace" + str(i) + "-yx.txt") as f:
            lines = [line.rstrip() for line in f]
    elif group == 7:
        with open("files/text/trace" + str(i) + "y-x.txt") as f:
            lines = [line.rstrip() for line in f]
    elif group == 8:
        with open("files/text/trace" + str(i) + "-y-x.txt") as f:
            lines = [line.rstrip() for line in f]

    for line in lines:
        x = line.split(", ")
        trace.append(x)

    return trace


group_traces()  # for text file generation
generate_images()  # for image file generation

