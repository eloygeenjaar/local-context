from glob import glob
import json

fileToWrite = '/data/users1/dkim195/local-global/ray_result_manual_2.txt'

run_paths = glob('*/')

num_layers_dic = {}
spatial_hidden_size_dic = {}
temporal_hidden_size_dic = {}
cnter = 0
with open(fileToWrite, 'a') as f:
    f.write("va_loss - epoch - config")
    f.write("\n")

for run_path in run_paths:

    metric_config_list = []

    trial_paths = glob(run_path + '*/')
    for trial_path in trial_paths:
        params_json = './' + trial_path + "params.json"
        params_json = open(params_json)
        params_json = json.load(params_json)

        result_csv = './' + trial_path + "progress.csv"

        with open(result_csv, "r", encoding="utf-8", errors="ignore") as scraped:
            final_line = scraped.readlines()[-1]
        
        final_line = final_line.split(',')
        va_loss = float(final_line[5])
        epoch = int(final_line[11])

        config = str(params_json)
        config = config[22:-1]
        config = eval(config)
        metric_config_list.append([va_loss, epoch, config ])

    metric_config_list.sort()
    metric_config_list = metric_config_list[:3]
    for line in metric_config_list:
        config = line[2]
        print(config)
        if 'num_layers' in config:
            num_layer = config['num_layers']
            num_layers_dic[num_layer] = num_layers_dic.get(num_layer, 0) + 1
            if num_layer == 4:
                cnter += 1
        if 'spatial_hidden_size' in config:
            spatial_hidden_size = config['spatial_hidden_size']
            spatial_hidden_size_dic[spatial_hidden_size] = spatial_hidden_size_dic.get(spatial_hidden_size, 0) + 1

        if 'temporal_hidden_size' in config:
            temporal_hidden_size = config['temporal_hidden_size']
            temporal_hidden_size_dic[temporal_hidden_size] = temporal_hidden_size_dic.get(temporal_hidden_size, 0) + 1

    with open(fileToWrite, 'a') as f:
        f.write("\n")
        f.write(run_path)
        f.write("\n")
        for line in metric_config_list:
            metric = str(line[0])
            epoch = str(line[1])
            config = str(line[2])
            f.write(metric)
            f.write(" - ")
            f.write(epoch)
            f.write(" - ")
            f.write(str(config))
            f.write('\n')

sorted_num_layers_dic = sorted(num_layers_dic.items(), key=lambda x:x[1], reverse = True)
sorted_spatial_hidden_size_dic = sorted(spatial_hidden_size_dic.items(), key=lambda x:x[1], reverse = True)
sorted_temporal_hidden_size_dic = sorted(temporal_hidden_size_dic.items(), key=lambda x:x[1], reverse = True)


with open(fileToWrite, 'a') as f:
    f.write('\n')
    f.write('num_layers_counter: ')
    f.write(str(sorted_num_layers_dic))

    f.write('\n')
    f.write('spatial_hidden_size_counter: ')
    f.write(str(sorted_spatial_hidden_size_dic))

    f.write('\n')
    f.write('temporal_hidden_size_counter: ')
    f.write(str(sorted_temporal_hidden_size_dic))


print(cnter)