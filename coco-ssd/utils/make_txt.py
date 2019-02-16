import json
from collections import OrderedDict

def make_txt(how_many_image):
    total_file = 40504
    test = 0.2
    validation = 0.25
    
    if how_many_image > total_file:
        print("Put number less than %d" %total_file)
        return
    elif how_many_image < 0:
        print("No negative number")
        return
    else:
        how_many_image = int(how_many_image)
    
    test_image = int(how_many_image * test)
    validation_image = int(how_many_image * validation)

    with open('/home/taeho/data/coco2014/annotations/annotations/instances_val2014.json', encoding="utf-8") as data_file:
        data = json.load(data_file, object_pairs_hook=OrderedDict)

    file_name = []
    for i in range(how_many_image):
        name = data["images"][i]["file_name"]
        name = name[:-4]
        file_name.append(name)

    # write
    f = open("/home/taeho/data/coco2014/images/test.txt", 'w')
    for i in file_name[:test_image]:
        f.write(i + "\n")
    f.close()
    f = open("/home/taeho/data/coco2014/images/val.txt", 'w')
    for i in file_name[test_image:(test_image+validation_image)]:
        f.write(i + "\n")
    f.close()
    f = open("/home/taeho/data/coco2014/images/train.txt", 'w')
    for i in file_name[(test_image+validation_image):]:
        f.write(i + "\n")
    f.close()
    
    print("test.txt:", len(file_name[:test_image]))
    print("val.txt:", len(file_name[test_image:(test_image+validation_image)]))
    print("train.txt:", len(file_name[(test_image+validation_image):]))
    return
