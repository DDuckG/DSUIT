import os
import argparse
import shutil
import traceback
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def preprocess(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    L = clahe.apply(l)
    Lab = cv2.merge([L, a, b])

    bgr = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gamma = 1.0
    if float(gray.mean()) < 40:
        gamma = 1.3
    elif float(gray.mean()) < 80:
        gamma = 1.1
    if gamma != 1.0:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype('uint8')
        gray = cv2.LUT(gray, table)
    return gray

def bright_percentile(gray, p = 10):
    return np.percentile(gray.flatten(), p)

def tenengrad_score(gray):
    fx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = 3)
    fy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = 3)
    fm = fx * fx + fy * fy
    return fm.mean()

def normalized_laplacian(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var = lap.var()
    mean = max(1.0, gray.mean())
    return var / (mean / 50)

def average_hash(gray, hash_size = 8):
    met = cv2.resize(gray, (hash_size, hash_size), interpolation = cv2.INTER_AREA)
    pxl = met.flatten().astype(np.uint8)
    avg = pxl.mean()
    bit = (pxl > avg).astype(np.uint8)
    bit_str = ''.join(bit.astype('str'))
    return bit_str

def hamming_diff(s1, s2):
    if s1 is None or s2 is None:
        return None
    len1, len2 = len(s1), len(s2)
    m = min(len1, len2)
    diff = sum(a != b for a, b in zip(s1[:m], s2[:m]))
    diff += abs(len1 - len2)
    return diff

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required = True, help = 'Source directory')
    p.add_argument('--out', required = True, help = 'Output directory for cleaned images')
    p.add_argument('--plot_hist', action='store_true', help = 'Save histogram')
    p.add_argument('--min_bright_pct', type = float, default = 10)
    p.add_argument('--min_tenengrad_scr', type = float, default = 80)
    p.add_argument('--min_norm_laplacian', type = float, default = 30)
    p.add_argument('--hash_size', type = int, default = 8)
    p.add_argument('--hamming_threshold', type = float, default = 5)
    p.add_argument('--min_mean_brightness', type = float, default=30.0)
    p.add_argument('--max_mean_brightness', type = float, default=230.0)
    p.add_argument('--verbose', action = 'store_true', help = 'Print action per file')
    arguments = p.parse_args()

    source = Path(arguments.src)
    output = Path(arguments.out)
    reject = output.parent / (output.name + '_rejected')
    low_light = output.parent / (output.name + '_low_light')

    output.mkdir(parents = True, exist_ok = True)
    reject.mkdir(parents = True, exist_ok = True)
    low_light.mkdir(parents = True, exist_ok = True)

    files = sorted([f for f in os.listdir(source) if f.lower().endswith('.jpg')])
    if not files:
        print(f'No images files in {source}')
        return

    hashes = []
    report = []
    keep_count = 0
    reject_count = 0
    low_light_count = 0

    for file in files:
        path = source / file
        if not path.is_file():
            continue
        
        reason = ''
        dupe_distance = ''
        try:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise IOError('cv2.imread returned None')
            
            gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean = gray_raw.mean()
            var = cv2.Laplacian(gray_raw, cv2.CV_64F).var()
            pct10 = bright_percentile(gray_raw, 10)

            gray = preprocess(img)
            tenengrad_scr = tenengrad_score(gray)
            norm_laplacian = normalized_laplacian(gray)

            # Dark img
            if pct10 < arguments.min_bright_pct:
                if (tenengrad_scr < arguments.min_tenengrad_scr) and (norm_laplacian < arguments.min_norm_laplacian):
                    reason = f"too_dark_and_blurry(pct10 = {pct10:.1f}, tenengrad = {tenengrad_scr:.1f}, normLaplacian = {norm_laplacian:.1f})"
                else:
                    aHash = average_hash(gray, hash_size = arguments.hash_size)
                    min_diff = None
                    is_dupe = False

                    for hash in hashes:
                        diff = hamming_diff(aHash, hash)
                        if diff is None:
                            continue
                        if min_diff is None or diff < min_diff:
                            min_diff = diff
                        if diff <= arguments.hamming_threshold:
                            dupe_distance = str(diff)
                            reason = f"duplicate (hamming = {diff} <= {arguments.hamming_threshold})"
                            is_dupe = True
                            break
                    
                    if is_dupe:
                        pass
                    else:
                        hashes.append(aHash)
                        dupe_distance = str(min_diff) if min_diff is not None else ''
                        shutil.copy2(str(path), str(low_light / file))
                        low_light_count += 1
                        status = 'keep'
                        report.append({'filename': file, 
                                       'var_raw': f"{var:.5f}", 
                                       'mean_raw': f"{mean:.5f}", 
                                       'pct10_raw': f"{pct10:.5f}", 
                                       'tenengrad_raw': f"{tenengrad_scr:.5f}", 
                                       'norm_laplacian_raw': f"{norm_laplacian:.5f}", 
                                       'status': status, 
                                       'reason': '', 
                                       'dupe_distance': dupe_distance})
                        
                        if arguments.verbose:
                            print('[KEEP_LOW_LIGHT]', file, f"pct10 = {pct10:.1f}, tenengrad_scr = {tenengrad_scr:.1f}")
                        continue
            else:
                if norm_laplacian < arguments.min_norm_laplacian:
                    reason = f"blurry (norm_laplacian = {norm_laplacian:.1f} < {arguments.min_norm_laplacian})"
                elif mean < arguments.min_mean_brightness:
                    reason = f"too dark (mean = {mean:.1f} < {arguments.min_mean_brightness})"
                elif mean > arguments.max_mean_brightness:
                    reason = f"too bright (mean = {mean:.1f} > {arguments.max_mean_brightness})"
            
            # Normal brightness img
            if reason == '':
                try:
                    aHash = average_hash(gray, hash_size = arguments.hash_size)
                except Exception:
                    aHash = None
                min_diff = None
                is_dupe = False
                for hash in hashes:
                    diff = hamming_diff(aHash, hash)
                    if diff is None:
                        continue
                    if min_diff is None or diff < min_diff:
                        min_diff = diff
                    if diff <= arguments.hamming_threshold:
                        dupe_distance = str(diff)
                        reason = f"duplicate (hamming = {diff} <= {arguments.hamming_threshold})"
                        is_dupe = True
                        break
                if not is_dupe:
                    hashes.append(aHash)
                    dupe_distance = str(min_diff) if min_diff is not None else ''
                    try:
                        shutil.copy2(str(path), str(output / file))
                        keep_count += 1
                        status = 'keep'
                        report.append({'filename': file, 
                                       'var_raw': f"{var:.5f}", 
                                       'mean_raw': f"{mean:.5f}", 
                                       'pct10_raw': f"{pct10:.5f}", 
                                       'tenengrad_raw': f"{tenengrad_scr:.5f}", 
                                       'norm_laplacian_raw': f"{norm_laplacian:.5f}", 
                                       'status': status, 
                                       'reason': '', 
                                       'dupe_distance': dupe_distance})
                        if arguments.verbose:
                            print('[KEEP]', file, f"normLaplacian = {norm_laplacian:.1f})", f"mean = {mean:.1f}")
                        continue
                    except Exception as error:
                        reason = f'copy_error: {error}'

            reject_count += 1
            try:
                shutil.move(str(path), str(reject / file))
            except Exception:
                try:
                    shutil.copy2(str(path), str(reject / file))
                    os.remove(str(path))
                except Exception:
                    pass
            if arguments.verbose:
                print('[REJECT]', file, reason) 
            report.append({'filename': file, 
                           'var_raw': f"{var:.5f}", 
                           'mean_raw': f"{mean:.5f}", 
                           'pct10_raw': f"{pct10:.5f}", 
                           'tenengrad_raw': f"{tenengrad_scr:.5f}", 
                           'norm_laplacian_raw': f"{norm_laplacian:.5f}", 
                           'status': 'reject', 
                           'reason': reason, 
                           'dupe_distance': dupe_distance})
            
        except Exception as error:
            traceb = traceback.format_exc()
            if arguments.verbose:
                print(f"[ERROR] processing {file}: {error}\n{traceb}")
            reject_count += 1
            try:
                shutil.move(str(path), str(reject / file))
            except Exception:
                pass
            report.append({'filename': file, 
                           'var_raw': '',
                           'mean_raw': '',
                           'pct10_raw': '',
                           'tenengrad_raw': '',
                           'norm_laplacian_raw': '',
                           'status': 'reject', 
                           'reason': 'processing_error', 
                           'dupe_distance': ''})
    
    report_file = output.parent / 'clean_report.csv'
    hist_file = output.parent / 'hist_data.csv'

    with open(report_file, 'w', newline='', encoding='utf-8') as rf:
        w = csv.writer(rf)
        header = ['filename', 'status', 'var_raw', 'mean_raw', 'pct10_raw', 'tenengrad_raw', 'norm_laplacian_raw', 'reason', 'dupe_distance']
        w.writerow(header)
        for r in report:
            w.writerow([r.get('filename',''), r.get('status',''), r.get('var_raw',''), r.get('mean_raw',''),
                        r.get('pct10_raw',''), r.get('tenengrad_raw',''), r.get('norm_laplacian_raw',''), r.get('reason',''), r.get('dupe_distance','')])

    with open(hist_file, 'w', newline='', encoding='utf-8') as hf:
        w = csv.writer(hf)
        w.writerow(['filename', 'var_raw', 'mean_raw', 'pct10_raw', 'tenengrad_raw', 'norm_laplacian_raw', 'status'])
        for r in report:
            w.writerow([r.get('filename',''), r.get('var_raw',''), r.get('mean_raw',''), r.get('pct10_raw',''),
                        r.get('tenengrad_raw',''), r.get('norm_laplacian_raw',''), r.get('status','')])
    
    print(f"Done. Kept={keep_count}, Kept_low_light={low_light_count}, Rejected={reject_count}. Report saved to {report_file}")

    if arguments.plot_hist:
        try:
            var_list = []
            mean_list = []
            pct10_list = []
            tenengrad_list = []
            normLaplacian_list = []
            for r in report:
                try:
                    if r.get('var_raw','') != '':
                        var_list.append(float(r.get('var_raw')))
                    if r.get('mean_raw','') != '':
                        mean_list.append(float(r.get('mean_raw')))
                    if r.get('pct10_raw','') != '':
                        pct10_list.append(float(r.get('pct10_raw')))
                    if r.get('tenengrad_raw','') != '':
                        tenengrad_list.append(float(r.get('tenengrad_raw')))
                    if r.get('norm_laplacian_raw','') != '':
                        normLaplacian_list.append(float(r.get('norm_laplacian_raw')))
                except Exception:
                    pass

            if var_list:
                plt.figure()
                plt.hist(var_list, bins = 50)
                plt.title('Raw Laplacian variance distribution')
                plt.xlabel('variance')
                plt.ylabel('count')
                plt.savefig(str(output.parent / 'var_hist.png'))
                plt.close()
            if mean_list:
                plt.figure()
                plt.hist(mean_list, bins = 50)
                plt.title('Raw mean brightness distribution')
                plt.xlabel('mean brightness')
                plt.ylabel('count')
                plt.savefig(str(output.parent / 'mean_hist.png'))
                plt.close()
            if pct10_list:
                plt.figure()
                plt.hist(pct10_list, bins = 50)
                plt.title('10th percentile brightness distribution (p10)')
                plt.xlabel('p10')
                plt.ylabel('count')
                plt.savefig(str(output.parent / 'pct10_hist.png'))
                plt.close()
            if tenengrad_list:
                plt.figure()
                plt.hist(tenengrad_list, bins = 50)
                plt.title('Tenengrad distribution')
                plt.xlabel('tenengrad')
                plt.ylabel('count')
                plt.savefig(str(output.parent / 'tenengrad_hist.png'))
                plt.close()
            if normLaplacian_list:
                plt.figure()
                plt.hist(normLaplacian_list, bins = 50)
                plt.title('Normalized Laplacian distribution')
                plt.xlabel('norm_laplacian')
                plt.ylabel('count')
                plt.savefig(str(output.parent / 'normlaplacian_hist.png'))
                plt.close()

        except Exception as error:
            print("[ERROR] while plotting hist: ", error)

if __name__ == '__main__':
    main()