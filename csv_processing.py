import csv
import os
import argparse
import numpy as np 

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--list', type=str, default = './reg_update_grid/cora')
parser.add_argument('--num', type=int, default = -2)
parser.add_argument('--csv', type=str, default = 'reg_update_grid_cora.csv')
args = parser.parse_args()

name = args.list
fileList = os.listdir(name)

def create_csv(csv_path):
    with open(csv_path,'w') as f:
        f.seek(0)
        f.truncate()
        csv_write = csv.writer(f)
        csv_head = ["sum", "acc", "std", "contrast_model", "hidden", "epochs", "lr", "wd", "reg", "ratio", "pn", "fn", "norm", "mu", "penalty", "pn-ratio", "layers", "noise struct", "noise ratio", "path"]
        # csv_head = [1,2,3,4,5]
        csv_write.writerow(csv_head)

create_csv(csv_path = args.csv)

for i, element in enumerate(fileList):
    print(element)
    file_path = os.path.join(name, element)
    # file_path = element
    sum = 0
    sum_v = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if reader.line_num == 2:
                contrast_model = row[0]
                hidden = row[1]
                epochs = row[2]
                lr = row[3] 
                weight_decay = row[4]
                reg = row[5] 
                ratio = row[6] 
                pre_norm = row[7]
                final_norm = row[8] 
                normalization = row[9] 
                mu = row[10] 
                norm_penalty = row[11] 
                penalty_ratio = row[12] 
                num_layers_to = row[13] 
                noise_struct = row[14] 
                modify_ratio = row[15]
            if reader.line_num > 2:
                print(row[args.num])
                sum+=float(row[args.num])
                sum_v.append(row[args.num])
    sum_v = list(map(float, sum_v))
    sum_v = np.array(sum_v)
    print(sum_v)
        
    with open(args.csv,'a+') as f:
        csv_write = csv.writer(f)
        data_row = [sum, sum_v.mean(), sum_v.std(), contrast_model, hidden, epochs, lr, weight_decay, reg, ratio, pre_norm, final_norm, normalization, mu, norm_penalty, penalty_ratio, num_layers_to, noise_struct, modify_ratio, file_path]
        csv_write.writerow(data_row)
    
