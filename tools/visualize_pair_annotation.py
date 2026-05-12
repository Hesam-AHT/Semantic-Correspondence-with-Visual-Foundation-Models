from PIL import Image, ImageDraw, ImageFont
import json
import glob
import os


kps_colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'white', 'gray', 'cyan',
              'red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'white', 'gray', 'cyan',
              'red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'white', 'gray', 'cyan']
kps_radius = 3


def drawrect(drawcontext, xy, color=None, width=3):
    x1, y1, x2, y2 = xy
    offset = 1
    for i in range(0, width):
        drawcontext.rectangle(((x1, y1), (x2, y2)), outline=color)
        x1 = x1 - offset
        y1 = y1 + offset
        x2 = x2 + offset
        y2 = y2 - offset


if __name__ == '__main__':
    proceed = input("\nWARNING: Visualizing SPair-71k dataset pairs takes a long time "
                    "and requires 7.2 GB of disk space.\n"
                    "Do you still want to proceed? (y/n): ")
    if not proceed == 'y':
        print('Visualization cancelled.')
        exit()

    pair_ann_path = '../PairAnnotation'
    img_path = '../JPEGImages'
    visual_path = '../Visualization/PairImages'

    print('\nVisualizing SPair-71k dataset pairs with their annotations...\n')

    if not os.path.exists(visual_path):
        os.makedirs(visual_path)

    pair_ann_subpaths = glob.glob('%s/*' % pair_ann_path)
    pair_ann_subpaths.sort()

    for pair_ann_subpath in pair_ann_subpaths:
        ann_paths = glob.glob('%s/*.json' % pair_ann_subpath)
        ann_paths.sort()
        split = os.path.basename(pair_ann_subpath)

        cur_visual_path = os.path.join(visual_path, split)
        if not os.path.exists(cur_visual_path):
            os.makedirs(cur_visual_path)

        for ann_path in ann_paths:
            with open(ann_path) as ann_file:
                ann = json.load(ann_file)

            filename = ann['filename']
            src_imname = ann['src_imname']
            trg_imname = ann['trg_imname']

            src_imsize = ann['src_imsize']
            trg_imsize = ann['trg_imsize']

            src_bndbox = ann['src_bndbox']
            trg_bndbox = ann['trg_bndbox']

            category = ann['category']

            cat_visual_path = os.path.join(cur_visual_path, category)
            if not os.path.exists(cat_visual_path):
                os.makedirs(cat_visual_path)

            src_pose = ann['src_pose']
            trg_pose = ann['trg_pose']

            src_kps = ann['src_kps']
            trg_kps = ann['trg_kps']
            assert len(src_kps) == len(trg_kps), 'The number of matching key-points NOT same'

            mirror = ann['mirror']
            viewpoint_variation = ann['viewpoint_variation']
            scale_variation = ann['scale_variation']
            truncation = ann['truncation']
            occlusion = ann['occlusion']

            src_img = Image.open(os.path.join(img_path, category, src_imname))
            trg_img = Image.open(os.path.join(img_path, category, trg_imname))
            src_draw = ImageDraw.Draw(src_img)
            trg_draw = ImageDraw.Draw(trg_img)

            drawrect(src_draw, (src_bndbox[0], src_bndbox[1], src_bndbox[2], src_bndbox[3]), 'red')
            drawrect(trg_draw, (trg_bndbox[0], trg_bndbox[1], trg_bndbox[2], trg_bndbox[3]), 'blue')

            assert len(src_kps) == len(trg_kps), 'Error'

            for kp_id, (src_kp, trg_kp) in enumerate(zip(src_kps, trg_kps)):
                if kp_id < 10:
                    src_draw.ellipse((src_kp[0] - kps_radius, src_kp[1] - kps_radius,
                                      src_kp[0] + kps_radius, src_kp[1] + kps_radius),
                                     fill=kps_colors[int(kp_id)], outline='black')
                    trg_draw.ellipse((trg_kp[0] - kps_radius, trg_kp[1] - kps_radius,
                                      trg_kp[0] + kps_radius, trg_kp[1] + kps_radius),
                                     fill=kps_colors[int(kp_id)], outline='black')
                elif kp_id < 20:
                    src_draw.rectangle((src_kp[0] - kps_radius, src_kp[1] - kps_radius,
                                        src_kp[0] + kps_radius, src_kp[1] + kps_radius),
                                       fill=kps_colors[int(kp_id)], outline='black')
                    trg_draw.rectangle((trg_kp[0] - kps_radius, trg_kp[1] - kps_radius,
                                        trg_kp[0] + kps_radius, trg_kp[1] + kps_radius),
                                       fill=kps_colors[int(kp_id)], outline='black')
                elif kp_id < 30:
                    src_draw.line((src_kp[0] - kps_radius, src_kp[1] - kps_radius,
                                   src_kp[0] + kps_radius, src_kp[1] + kps_radius),
                                  fill=kps_colors[int(kp_id)], width=2)
                    src_draw.line((src_kp[0] + kps_radius, src_kp[1] - kps_radius,
                                   src_kp[0] - kps_radius, src_kp[1] + kps_radius),
                                  fill=kps_colors[int(kp_id)], width=2)
                    trg_draw.line((trg_kp[0] - kps_radius, trg_kp[1] - kps_radius,
                                   trg_kp[0] + kps_radius, trg_kp[1] + kps_radius),
                                  fill=kps_colors[int(kp_id)], width=2)
                    trg_draw.line((trg_kp[0] + kps_radius, trg_kp[1] - kps_radius,
                                   trg_kp[0] - kps_radius, trg_kp[1] + kps_radius),
                                  fill=kps_colors[int(kp_id)], width=2)

            total_width = src_imsize[0] + trg_imsize[0]
            desc_height = 230
            total_height = max(src_imsize[1], trg_imsize[1]) + desc_height
            des_img = Image.new("RGB", (total_width, desc_height), color='black')

            mirror_str = 'True' if mirror else 'False'
            if viewpoint_variation == 0:
                viewpoint_variation_str = 'Easy'
            elif viewpoint_variation == 1:
                viewpoint_variation_str = 'Medium'
            elif viewpoint_variation == 2:
                viewpoint_variation_str = 'Difficult'

            if scale_variation == 0:
                scale_variation_str = 'Easy'
            elif scale_variation == 1:
                scale_variation_str = 'Medium'
            elif scale_variation == 2:
                scale_variation_str = 'Difficult'

            if truncation == 0:
                truncation_str = 'None'
            elif truncation == 1:
                truncation_str = 'Source Truncated'
            elif truncation == 2:
                truncation_str = 'Target Truncated'
            elif truncation == 3:
                truncation_str = 'Both Truncated'

            if occlusion == 0:
                occlusion_str = 'None'
            elif occlusion == 1:
                occlusion_str = 'Source Occluded'
            elif occlusion == 2:
                occlusion_str = 'Target Occluded'
            elif occlusion == 3:
                occlusion_str = 'Both Occluded'


            fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 20)
            des_draw = ImageDraw.Draw(des_img)
            des_draw.text((10, 10), filename, font=fnt, fill='white')
            des_draw.text((10, 35), 'Mirror: %s (%d)' % (mirror_str, mirror), font=fnt, fill='red')
            des_draw.text((10, 60), 'VP-var: %s (%d)' % (viewpoint_variation_str, viewpoint_variation), font=fnt, fill='yellow')
            des_draw.text((10, 85), 'SC-var: %s (%d)' % (scale_variation_str, scale_variation), font=fnt, fill='cyan')
            des_draw.text((10, 110), 'Truncn: %s (%d)' % (truncation_str, truncation), font=fnt, fill='pink')
            des_draw.text((10, 135), 'Occlun: %s (%d)' % (occlusion_str, occlusion), font=fnt, fill='gray')
            des_draw.text((10, 160), 'Num sharing kps: %d' % len(src_kps), font=fnt, fill='white')

            new_im = Image.new('RGB', (total_width, total_height))
            new_im.paste(src_img, (0, 0))
            new_im.paste(trg_img, (src_imsize[0], 0))
            new_im.paste(des_img, (0, max(src_imsize[1], trg_imsize[1])))
            new_im_draw = ImageDraw.Draw(new_im)

            for kp_id, (src_kp, trg_kp) in enumerate(zip(src_kps, trg_kps)):
                new_im_draw.line((src_kp[0], src_kp[1], trg_kp[0] + src_imsize[0], trg_kp[1]),
                                 fill=kps_colors[int(kp_id)], width=2)

            trg_save_path = os.path.join(visual_path, split, category, filename) + '.jpg'
            new_im.save(trg_save_path)

            print('\tSaving results_SPair71K:  %s' % trg_save_path)

    print('\nPair visualization finished in %s.' % visual_path)
