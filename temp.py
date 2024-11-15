

out = 'out.txt'
cl = 'clipart_list.txt'
pl = 'painting_list.txt'
rl = 'real_list.txt'
sl = 'sketch_list.txt'

all_cls = set()
with open(out, 'r') as f:
    for line in f.read().split('\n'):
        all_cls.add(line)

print('All', len(all_cls))

cl_cls = set()
with open(cl, 'r') as f1:
    for line in f1:
        cur_cls = line.split('/')[1]
        if cur_cls not in all_cls:
            print(cur_cls)
        cl_cls.add(cur_cls)

print('Cl', len(cl_cls))

pl_cls = set()
with open(pl, 'r') as f2:
    for line in f2:
        cur_cls = line.split('/')[1]
        if cur_cls not in all_cls:
            print(cur_cls)
        pl_cls.add(cur_cls)

print('Pl', len(pl_cls))

rl_cls = set()
with open(rl, 'r') as f1:
    for line in f1:
        cur_cls = line.split('/')[1]
        if cur_cls not in all_cls:
            print(cur_cls)
        rl_cls.add(cur_cls)

print('Rl', len(rl_cls))

sl_cls = set()
with open(sl, 'r') as f1:
    for line in f1:
        cur_cls = line.split('/')[1]
        if cur_cls not in all_cls:
            print(cur_cls)
        sl_cls.add(cur_cls)

print('Sl', len(sl_cls))

print('All 4', len(cl_cls | pl_cls | rl_cls | sl_cls))

print(all_cls - cl_cls)