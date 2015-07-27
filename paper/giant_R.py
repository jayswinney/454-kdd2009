import os

my_fp = 'c:/Users/Jay/Dropbox/pred_454_team/'


def find_r_script(fp):
    out = []
    if fp == 'c:/Users/Jay/Dropbox/pred_454_team/paper':
        out.append(
            'c:/Users/Jay/Dropbox/pred_454_team/paper/script_for_paper.r')
    for f in os.listdir(fp):
        if os.path.isdir(fp + f):
            out.extend(find_r_script(fp + f + '/'))
        elif f[-2:].lower() == '.r':
            out.append(fp + f)

    return out


r_scripts = find_r_script(my_fp)

giant_r = open(
    'c:/Users/Jay/Dropbox/pred_454_team/paper/giant.r', 'wb')

giant_r.write('# ---- giant r ----')
giant_r.write('\n')

for f in r_scripts:
    giant_r.write('#' + f.split('/')[-1])
    giant_r.write('\n')
    with open(f, 'rb') as script:
        s = script.read()
        s = s.replace('# ----', '# _____')
        s = s.replace('#----', '# _____')
        giant_r.write(s)

    giant_r.write('\n')
    giant_r.write('#_______________________________________________________')
    giant_r.write('\n')
