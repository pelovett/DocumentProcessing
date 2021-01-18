import os
import math

DATA_DIR = set(['cs_AI', 'cs_CV', 'cs_IT', 'cs_PL', 'math_AC',
                'math_ST', 'cs_CE', 'cs_DS', 'cs_NE', 'cs_SY', 'math_GR'])


def split_files():
    os.mkdir('./arxiv/train/')
    os.mkdir('./arxiv/validation/')
    os.mkdir('./arxiv/test/')

    for dirname in os.listdir('./arxiv/'):
        dir_path = './arxiv/'+dirname
        if dirname not in DATA_DIR:
            continue
        os.mkdir('./arxiv/train/'+dirname)
        os.mkdir('./arxiv/validation/'+dirname)
        os.mkdir('./arxiv/test/'+dirname)
        x = []
        for paper_file in os.listdir(dir_path):
            with open(dir_path+'/'+paper_file, 'r') as in_file:
                x.append((paper_file, in_file.read()))
        x = sorted(x, key=lambda y: y[1])  # Sort for consistent ordering
        train_end = math.floor(len(x)*.7)
        val_end = math.floor((len(x)-train_end)*.666)
        for i in range(0, train_end):
            with open('./arxiv/train/'+dirname+'/'+x[i][0], 'w') as out_file:
                out_file.write(x[i][1])
        for i in range(train_end, train_end+val_end):
            with open('./arxiv/validation/'+dirname+'/'+x[i][0], 'w') as out_file:
                out_file.write(x[i][1])
        for i in range(train_end+val_end, len(x)):
            with open('./arxiv/test/'+dirname+'/'+x[i][0], 'w') as out_file:
                out_file.write(x[i][1])
        print(f'Finished dir: {dirname}')


if __name__ == "__main__":
    split_files()
