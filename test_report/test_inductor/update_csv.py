import os

prefix_str = "inductor/"

for root, dirs, files in os.walk("./"):
    for file in files:
        if ".csv" in file: 
            f_p = os.path.join(root, file)
            with open(f_p) as f:
                lines = f.readlines()
                new_lines = []
                for line in lines:
                    if line != "" and line != "\n" and "test_filename" not in line and prefix_str not in line:
                        line = prefix_str + line
                    if ",failed," in line and "#" not in line and line[0] != " ":
                        line = "# " + line
                    new_lines.append(line)

            with open(f_p, "w") as f:
                f.writelines(new_lines)


