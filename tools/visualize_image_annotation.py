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
    img_ann_path = '../ImageAnnotation'
    img_jpg_path = '../JPEGImages'
    visual_path = '../Visualization'

    print('Visualizing SPair-71k dataset images with their annotations...\n')

    img_ann_subpaths = glob.glob('%s/*' % img_ann_path)
    img_jpg_subpaths = glob.glob('%s/*' % img_jpg_path)
    img_ann_subpaths.sort()
    img_jpg_subpaths.sort()

    visual_path = os.path.join(visual_path, 'JPEGImages')
    if not os.path.exists(visual_path):
        os.makedirs(visual_path)

    for img_ann_subpath, img_jpg_subpath in zip(img_ann_subpaths, img_jpg_subpaths):
        img_ann_cat = os.path.basename(img_ann_subpath)
        img_jpg_cat = os.path.basename(img_jpg_subpath)

        assert img_ann_cat == img_jpg_cat, 'Category not matching.'
        cat = img_ann_cat
        print('\n\tVisualizing images of category %s...' % cat)

        cur_visual_path = os.path.join(visual_path, cat)

        if not os.path.exists(cur_visual_path):
            os.makedirs(cur_visual_path)

        img_anns = glob.glob('%s/*.json' % img_ann_subpath)
        img_jpgs = glob.glob('%s/*.jpg' % img_jpg_subpath)
        img_anns.sort()
        img_jpgs.sort()

        for img_ann, img_jpg in zip(img_anns, img_jpgs):
            img_ann_id = os.path.basename(img_ann).split('.')[0]
            img_jpg_id = os.path.basename(img_jpg).split('.')[0]

            assert img_ann_id == img_jpg_id, 'Image and Annotation not matching.'
            img_id = img_ann_id
            final_visual_path = os.path.join(cur_visual_path, img_id + '.jpg')

            with open(img_ann) as f:
                ann = json.load(f)
            kps = ann['kps']
            bbox = ann['bndbox']
            assert len(kps) == 30, 'len(kps) Error.'

            filename = ann['filename']
            src_database = ann['src_database']
            src_annotation = ann['src_annotation']
            src_image = ann['src_image']
            image_width = ann['image_width']
            image_height = ann['image_height']
            image_depth = ann['image_depth']
            category = ann['category']
            pose = ann['pose']
            truncated = ann['truncated']
            occluded = ann['occluded']
            difficult = ann['difficult']
            azimuth_id = ann['azimuth_id']
            assert img_id == filename.split('.')[0], 'Image and Id in annotation file not matching.'
            assert category == cat, 'Category not matching.'

            nkps = 0
            for kp_id in kps:
                if kps[kp_id] is not None:
                    nkps += 1

            img_obj = Image.open(img_jpg)
            kps_img_obj = Image.open(img_jpg)
            box_img_obj = Image.open(img_jpg)
            des_img_obj = Image.new("RGB", box_img_obj.size, color='black')

            images = [img_obj, kps_img_obj, box_img_obj, des_img_obj]
            widths, heights = zip(*(i.size for i in images))

            total_width = sum(widths) // 2
            total_height = sum(heights) // 2

            new_im = Image.new('RGB', (total_width, total_height))

            # Draw key-points
            for keypoint_id in kps:
                keypoint = kps[keypoint_id]

                if keypoint is None:
                    continue
                else:
                    draw = ImageDraw.Draw(kps_img_obj)
                    if int(keypoint_id) < 10:
                        draw.ellipse((keypoint[0] - kps_radius, keypoint[1] - kps_radius,
                                      keypoint[0] + kps_radius, keypoint[1] + kps_radius),
                                     fill=kps_colors[int(keypoint_id)], outline='black')
                    elif int(keypoint_id) < 20:
                        draw.rectangle((keypoint[0] - kps_radius, keypoint[1] - kps_radius,
                                        keypoint[0] + kps_radius, keypoint[1] + kps_radius),
                                       fill=kps_colors[int(keypoint_id)], outline='black')
                    elif int(keypoint_id) < 30:
                        draw.line((keypoint[0] - kps_radius, keypoint[1] - kps_radius,
                                   keypoint[0] + kps_radius, keypoint[1] + kps_radius),
                                  fill=kps_colors[int(keypoint_id)], width=2)
                        draw.line((keypoint[0] + kps_radius, keypoint[1] - kps_radius,
                                   keypoint[0] - kps_radius, keypoint[1] + kps_radius),
                                  fill=kps_colors[int(keypoint_id)], width=2)

            # Draw bounding-box
            draw = ImageDraw.Draw(box_img_obj)
            drawrect(draw, (bbox[0], bbox[1], bbox[2], bbox[3]), color='red', width=3)

            # Draw descriptions
            fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 15)
            fnt2 = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 20)
            draw = ImageDraw.Draw(des_img_obj)
            draw.text((10, 10), filename, font=fnt, fill='white')
            draw.text((10, 30), '(%d, %d, %d)' % (image_width, image_height, image_depth), font=fnt, fill='white')
            draw.text((10, 50), '%s %s' % (category, pose), font=fnt, fill='white')

            if truncated:
                draw.text((10, 75), 'Truncated', font=fnt2, fill='yellow')
            if occluded:
                draw.text((10, 100), 'Occluded', font=fnt2, fill='red')
            if difficult:
                draw.text((10, 125), 'Difficult', font=fnt2, fill='cyan')

            draw.text((10, 150), 'Azimuth_id: %d' % azimuth_id, font=fnt2, fill='white')
            draw.text((10, 175), 'Num Keypoints: %d' % nkps, font=fnt2, fill='white')

            x_offset = 0
            y_offset = 0
            for idx, im in enumerate(images):
                new_im.paste(im, (x_offset, y_offset))
                if idx == 0 or idx == 2:
                    x_offset += im.size[0]
                elif idx == 1:
                    x_offset = 0
                    y_offset += im.size[1]

            new_im.save(final_visual_path)

            print('\tSaving results_SPair71K:  %s' % final_visual_path)

    print('\nImage visualization finished in %s.' % visual_path)
