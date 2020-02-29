def read_asf(file_name):
    asf = open(file_name, "r")
    lines = asf.readlines()
    num_info = int(lines[9])
    final_lines = []
    for i in range(16, num_info + 16):
        final_lines.append(lines[i])
    return final_lines

path = "/home/dewey/Misc/db/26-6m.asf"
read_asf(path)
