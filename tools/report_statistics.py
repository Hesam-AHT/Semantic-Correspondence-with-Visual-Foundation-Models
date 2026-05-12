import matplotlib.pyplot as plt
import numpy as np
import json
import glob
import os


def draw_circular_histogram(vpvar_stats, cats, savepath):
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'gray']
    theta = np.linspace(22.5 * (np.pi / 180), 2 * np.pi + (22.5 * (np.pi / 180)), 8, endpoint=False)
    width = (2 * np.pi) / 8
    for trn_vpvar_stat, cat in zip(vpvar_stats, cats):
        radii = trn_vpvar_stat[:8]
        ax = plt.subplot(111, polar=True)
        title = ax.set_title(cat)
        title.set_position([0.5, 1.08])
        bars = ax.bar(theta, radii, width=width, bottom=0)

        for r, bar, color in zip(list(range(18)), bars, colors):
            bar.set_facecolor(color)
            bar.set_alpha(0.8)

        plt.savefig(os.path.join(savepath, cat + '_histogram.png'))
        plt.close()
    return


def draw_vpvar_table(vpvar_stats, cats, columns, datatype, savepath):
    fig, ax = plt.subplots(1, 1)
    plt.subplots_adjust(left=0.2, top=0.2)
    ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(cellText=vpvar_stats,
                         rowLabels=cats,
                         colLabels=columns,
                         colLoc='center',
                         loc='top')
    title = ax.set_title('%sSet Azimuth Distribution Table' % datatype)
    title.set_position([0.4, 9])
    plt.savefig('%s/../%sSet_Statistic.png' % (savepath, datatype))
    return


def trnvaltest_images(layout_path):
    trn_file = os.path.join(layout_path, 'trn.txt')
    val_file = os.path.join(layout_path, 'val.txt')
    test_file = os.path.join(layout_path, 'test.txt')

    trn_lines = open(trn_file).read().split('\n')
    val_lines = open(val_file).read().split('\n')
    test_lines = open(test_file).read().split('\n')
    trn_lines = trn_lines[:len(trn_lines)-1]
    val_lines = val_lines[:len(val_lines)-1]
    test_lines = test_lines[:len(test_lines)-1]

    total_imgs = dict()
    trn_imgs = dict()
    val_imgs = dict()
    test_imgs = dict()

    for trn_line in trn_lines:
        src_img_id = trn_line.split(':')[0].split('-')[1]
        trg_img_id = trn_line.split(':')[0].split('-')[2]
        src_img_id = src_img_id + ':' + trn_line.split(':')[1]
        trg_img_id = trg_img_id + ':' + trn_line.split(':')[1]

        if trn_imgs.get(src_img_id) is None:
            trn_imgs[src_img_id] = 0
        if trn_imgs.get(trg_img_id) is None:
            trn_imgs[trg_img_id] = 0
        if total_imgs.get(src_img_id) is None:
            total_imgs[src_img_id] = 0
        if trn_imgs.get(trg_img_id) is None:
            total_imgs[trg_img_id] = 0

    for val_line in val_lines:
        src_img_id = val_line.split(':')[0].split('-')[1]
        trg_img_id = val_line.split(':')[0].split('-')[2]
        src_img_id = src_img_id + ':' + val_line.split(':')[1]
        trg_img_id = trg_img_id + ':' + val_line.split(':')[1]

        if val_imgs.get(src_img_id) is None:
            val_imgs[src_img_id] = 0
        if val_imgs.get(trg_img_id) is None:
            val_imgs[trg_img_id] = 0
        if total_imgs.get(src_img_id) is None:
            total_imgs[src_img_id] = 0
        if trn_imgs.get(trg_img_id) is None:
            total_imgs[trg_img_id] = 0

    for test_line in test_lines:
        src_img_id = test_line.split(':')[0].split('-')[1]
        trg_img_id = test_line.split(':')[0].split('-')[2]
        src_img_id = src_img_id + ':' + test_line.split(':')[1]
        trg_img_id = trg_img_id + ':' + test_line.split(':')[1]

        if test_imgs.get(src_img_id) is None:
            test_imgs[src_img_id] = 0
        if test_imgs.get(trg_img_id) is None:
            test_imgs[trg_img_id] = 0
        if total_imgs.get(src_img_id) is None:
            total_imgs[src_img_id] = 0
        if trn_imgs.get(trg_img_id) is None:
            total_imgs[trg_img_id] = 0

    return total_imgs, trn_imgs, val_imgs, test_imgs


def image_statistics(img_ann_path, imgstat_save_path, total_dict, trn_dict, val_dict, test_dict):
    img_ann_subpaths = glob.glob('%s/*' % img_ann_path)
    img_ann_subpaths.sort()

    trn_stats = dict()
    val_stats = dict()
    test_stats = dict()

    for img_ann_subpath in img_ann_subpaths:
        img_anns = glob.glob('%s/*.json' % img_ann_subpath)
        img_anns.sort()

        for img_ann in img_anns:
            with open(img_ann) as ann_file:
                ann = json.load(ann_file)

            category = ann['category']
            img_id = ann['filename'].split('.')[0]
            img_id = img_id + ':' + category
            truncated = ann['truncated']
            occluded = ann['occluded']
            azimuth_id = ann['azimuth_id']

            if trn_stats.get(category) is None:
                trn_stats[category] = [[] for i in range(8)]
            if val_stats.get(category) is None:
                val_stats[category] = [[] for i in range(8)]
            if test_stats.get(category) is None:
                test_stats[category] = [[] for i in range(8)]

            if trn_dict.get(img_id) is not None:
                trn_stats[category][azimuth_id].append(img_id)
            elif val_dict.get(img_id) is not None:
                val_stats[category][azimuth_id].append(img_id)
            elif test_dict.get(img_id) is not None:
                test_stats[category][azimuth_id].append(img_id)
            else:
                raise Exception('Unexpected error.')

    cats = []
    trn_vpvar_stats = [[] for i in range(18)]
    for idx, cat in enumerate(trn_stats):
        cats.append(cat)
        catsum = 0
        for azm_cat in trn_stats[cat]:
            trn_vpvar_stats[idx].append(len(azm_cat))
            catsum += len(azm_cat)
        trn_vpvar_stats[idx].append(catsum)

    val_vpvar_stats = [[] for i in range(18)]
    for idx, cat in enumerate(val_stats):
        catsum = 0
        for azm_cat in val_stats[cat]:
            val_vpvar_stats[idx].append(len(azm_cat))
            catsum += len(azm_cat)
        val_vpvar_stats[idx].append(catsum)

    test_vpvar_stats = [[] for i in range(18)]
    for idx, cat in enumerate(test_stats):
        catsum = 0
        for azm_cat in test_stats[cat]:
            test_vpvar_stats[idx].append(len(azm_cat))
            catsum += len(azm_cat)
        test_vpvar_stats[idx].append(catsum)

    columns = list(range(8))
    columns.append('Total')
    cats.sort()
    cats.append('Total')

    trn_vpvar_stats.append(list(np.sum(trn_vpvar_stats, axis=0)))
    trn_vpvar_stats = np.array(trn_vpvar_stats)
    val_vpvar_stats.append(list(np.sum(val_vpvar_stats, axis=0)))
    val_vpvar_stats = np.array(val_vpvar_stats)
    test_vpvar_stats.append(list(np.sum(test_vpvar_stats, axis=0)))
    test_vpvar_stats = np.array(test_vpvar_stats)

    if not os.path.exists(os.path.join(imgstat_save_path, 'trn')):
        os.makedirs(os.path.join(imgstat_save_path, 'trn'))
    if not os.path.exists(os.path.join(imgstat_save_path, 'val')):
        os.makedirs(os.path.join(imgstat_save_path, 'val'))
    if not os.path.exists(os.path.join(imgstat_save_path, 'test')):
        os.makedirs(os.path.join(imgstat_save_path, 'test'))
    if not os.path.exists(os.path.join(imgstat_save_path, 'all')):
        os.makedirs(os.path.join(imgstat_save_path, 'all'))

    all_vpvar_stats = trn_vpvar_stats + val_vpvar_stats + test_vpvar_stats

    draw_circular_histogram(trn_vpvar_stats, cats, os.path.join(imgstat_save_path, 'trn'))
    draw_circular_histogram(val_vpvar_stats, cats, os.path.join(imgstat_save_path, 'val'))
    draw_circular_histogram(test_vpvar_stats, cats, os.path.join(imgstat_save_path, 'test'))
    draw_circular_histogram(all_vpvar_stats, cats, os.path.join(imgstat_save_path, 'all'))

    draw_vpvar_table(trn_vpvar_stats, cats, columns, 'Train', os.path.join(imgstat_save_path, 'trn'))
    draw_vpvar_table(val_vpvar_stats, cats, columns, 'Val', os.path.join(imgstat_save_path, 'val'))
    draw_vpvar_table(test_vpvar_stats, cats, columns, 'Test', os.path.join(imgstat_save_path, 'test'))
    draw_vpvar_table(all_vpvar_stats, cats, columns, 'All', os.path.join(imgstat_save_path, 'all'))

    return


def count_challenges(pairs, pairann_path):
    vpvar_cnt = [0, 0, 0]
    scvar_cnt = [0, 0, 0]
    trncn_cnt = [0, 0, 0, 0]
    occln_cnt = [0, 0, 0, 0]

    cat_pair_cnt = dict()

    for pair in pairs:
        ann_file = os.path.join(pairann_path, pair) + '.json'
        with open(ann_file) as f:
            ann = json.load(f)

        if cat_pair_cnt.get(ann['category']) is None:
            cat_pair_cnt[ann['category']] = 0
        cat_pair_cnt[ann['category']] += 1

        vpvar_cnt[ann['viewpoint_variation']] += 1
        scvar_cnt[ann['scale_variation']] += 1
        trncn_cnt[ann['truncation']] += 1
        occln_cnt[ann['occlusion']] += 1

    return vpvar_cnt, scvar_cnt, trncn_cnt, occln_cnt, cat_pair_cnt


def pair_statistics(pairann_path, layout_path, pairstat_save_path, splittype):
    trn_filepath = os.path.join(layout_path, splittype, 'trn.txt')
    val_filepath = os.path.join(layout_path, splittype, 'val.txt')
    test_filepath = os.path.join(layout_path, splittype, 'test.txt')

    trn_pairs = open(trn_filepath).read().split('\n')
    val_pairs = open(val_filepath).read().split('\n')
    test_pairs = open(test_filepath).read().split('\n')
    trn_pairs = trn_pairs[:len(trn_pairs)-1]
    val_pairs = val_pairs[:len(val_pairs) - 1]
    test_pairs = test_pairs[:len(test_pairs) - 1]

    trn_pairann_path = os.path.join(pairann_path, 'trn')
    val_pairann_path = os.path.join(pairann_path, 'val')
    test_pairann_path = os.path.join(pairann_path, 'test')

    trn_vpvar_cnt, trn_scvar_cnt, trn_trncn_cnt, trn_occln_cnt, trn_cat_pair_cnt = count_challenges(trn_pairs, trn_pairann_path)
    val_vpvar_cnt, val_scvar_cnt, val_trncn_cnt, val_occln_cnt, val_cat_pair_cnt = count_challenges(val_pairs, val_pairann_path)
    test_vpvar_cnt, test_scvar_cnt, test_trncn_cnt, test_occln_cnt, test_cat_pair_cnt = count_challenges(test_pairs, test_pairann_path)
    all_vpvar_cnt = [sum(x) for x in zip(trn_vpvar_cnt, val_vpvar_cnt, test_vpvar_cnt)]
    all_scvar_cnt = [sum(x) for x in zip(trn_scvar_cnt, val_scvar_cnt, test_scvar_cnt)]
    all_trncn_cnt = [sum(x) for x in zip(trn_trncn_cnt, val_trncn_cnt, test_trncn_cnt)]
    all_occln_cnt = [sum(x) for x in zip(trn_occln_cnt, val_occln_cnt, test_occln_cnt)]

    all_cat_pair_cnt = dict()
    for cat in trn_cat_pair_cnt:
        all_cat_pair_cnt[cat] = trn_cat_pair_cnt[cat] + val_cat_pair_cnt[cat] + test_cat_pair_cnt[cat]

    result_str = '\n'
    result_str += '+================+==  trn ==+==  val ==+== test ==+==  all ==+\n'
    for cat in trn_cat_pair_cnt:
        result_str += '| %14s | %8d | %8d | %8d | %8d |\n' % \
              (cat, trn_cat_pair_cnt[cat], val_cat_pair_cnt[cat], test_cat_pair_cnt[cat], all_cat_pair_cnt[cat])
    result_str += '+================+==========+==========+==========+==========+\n'
    result_str += '| %14s | %8d | %8d | %8d | %8d |\n' % \
                  ('all',
                   sum(list(map(lambda x: trn_cat_pair_cnt[x], trn_cat_pair_cnt))),
                   sum(list(map(lambda x: val_cat_pair_cnt[x], val_cat_pair_cnt))),
                   sum(list(map(lambda x: test_cat_pair_cnt[x], test_cat_pair_cnt))),
                   sum(list(map(lambda x: all_cat_pair_cnt[x], test_cat_pair_cnt))))
    result_str += '+================+==========+==========+==========+==========+\n'

    result_str += '\n'
    result_str += '+------+-----------------------------------------------------------------------------------------------------------------------------+\n'
    result_str += '|      |      View-point Var.     |        Scale Var.        |             Truncation            |             Occlusion             |\n'
    result_str += '|      |  Easy  | Medium | Diffi. |  Easy  | Medium | Diffi. |  None  | Source | Target |  Both  |  None  | Source | Target |  Both  |\n'
    result_str += '+------+-----------------------------------------------------------------------------------------------------------------------------+\n'
    result_str += '| trn  | %6d | %6d | %6d | %6d | %6d | %6d | %6d | %6d | %6d | %6d | %6d | %6d | %6d | %6d |\n' % \
                  (trn_vpvar_cnt[0], trn_vpvar_cnt[1], trn_vpvar_cnt[2], trn_scvar_cnt[0], trn_scvar_cnt[1], trn_scvar_cnt[2],
                   trn_trncn_cnt[0], trn_trncn_cnt[1], trn_trncn_cnt[2], trn_trncn_cnt[3],
                   trn_occln_cnt[0], trn_occln_cnt[1], trn_occln_cnt[2], trn_occln_cnt[3])
    result_str += '| val  | %6d | %6d | %6d | %6d | %6d | %6d | %6d | %6d | %6d | %6d | %6d | %6d | %6d | %6d |\n' % \
          (val_vpvar_cnt[0], val_vpvar_cnt[1], val_vpvar_cnt[2], val_scvar_cnt[0], val_scvar_cnt[1], val_scvar_cnt[2],
           val_trncn_cnt[0], val_trncn_cnt[1], val_trncn_cnt[2], val_trncn_cnt[3],
           val_occln_cnt[0], val_occln_cnt[1], val_occln_cnt[2], val_occln_cnt[3])
    result_str += '| test | %6d | %6d | %6d | %6d | %6d | %6d | %6d | %6d | %6d | %6d | %6d | %6d | %6d | %6d |\n' % \
          (test_vpvar_cnt[0], test_vpvar_cnt[1], test_vpvar_cnt[2], test_scvar_cnt[0], test_scvar_cnt[1], test_scvar_cnt[2],
           test_trncn_cnt[0], test_trncn_cnt[1], test_trncn_cnt[2], test_trncn_cnt[3],
           test_occln_cnt[0], test_occln_cnt[1], test_occln_cnt[2], test_occln_cnt[3])
    result_str += '+------+-----------------------------------------------------------------------------------------------------------------------------+\n'
    result_str += '| all  | %6d | %6d | %6d | %6d | %6d | %6d | %6d | %6d | %6d | %6d | %6d | %6d | %6d | %6d |\n' % \
          (all_vpvar_cnt[0], all_vpvar_cnt[1], all_vpvar_cnt[2], all_scvar_cnt[0], all_scvar_cnt[1], all_scvar_cnt[2],
           all_trncn_cnt[0], all_trncn_cnt[1], all_trncn_cnt[2], all_trncn_cnt[3],
           all_occln_cnt[0], all_occln_cnt[1], all_occln_cnt[2], all_occln_cnt[3])
    result_str += '+------+-----------------------------------------------------------------------------------------------------------------------------+\n'

    file = open(pairstat_save_path + '.txt', 'w')
    file.write(result_str)

    return


if __name__ == '__main__':
    img_ann_path = '../ImageAnnotation'
    img_jpg_path = '../JPEGImages'
    pairann_path = '../PairAnnotation'
    layout_path = '../Layout'
    stat_save_path = '../Visualization/Statistics'
    imgstat_save_path = os.path.join(stat_save_path, 'images')
    pairstat_save_path = os.path.join(stat_save_path, 'pairs')

    print('\nGenerating SPair-71k dataset statistics...\n')

    if not os.path.exists(stat_save_path):
        os.makedirs(stat_save_path)
    if not os.path.exists(imgstat_save_path):
        os.makedirs(imgstat_save_path)
    if not os.path.exists(pairstat_save_path):
        os.makedirs(pairstat_save_path)

    totalL, trnL, valL, testL = trnvaltest_images('../Layout/large')
    totalS, trnS, valS, testS = trnvaltest_images('../Layout/small')

    image_statistics(img_ann_path, imgstat_save_path, totalL, trnL, valL, testL)
    print('\tImage statistic generated.')

    pairstatL_save_path = os.path.join(pairstat_save_path, 'large')
    pairstatS_save_path = os.path.join(pairstat_save_path, 'small')

    pair_statistics(pairann_path, layout_path, pairstatL_save_path, 'large')
    print('\tImage statistic (Large) generated.')

    pair_statistics(pairann_path, layout_path, pairstatS_save_path, 'small')
    print('\tImage statistic (Small) generated.')

    print('\nSPair-71k dataset statistics generation finished.\n')
