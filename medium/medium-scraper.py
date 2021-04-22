

class

def convert_day(day):
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    m = 0
    d = 0
    while day > 0:
        d = day
        day -= month_days[m]
        m += 1
    return (m, d)

def get_claps(claps_str):
    if (claps_str is None) or (claps_str == '') or (claps_str.split is None):
        return 0
    split = claps_str.split('K')
    claps = float(split[0])
    claps = int(claps*1000) if len(split) == 2 else int(claps)
    return claps

def get_img(img_url, dest_folder, dest_filename):
    ext = img_url.split('.')[-1]
    if len(ext) > 4:
        ext = 'jpg'
    dest_filename = f'{dest_filename}.{ext}'
    with open(f'{dest_folder}/{dest_filename}', 'wb') as f:
        f.write(requests.get(img_url, allow_redirects=False).content)
    return dest_filename