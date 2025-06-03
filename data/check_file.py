import json

# Mapping lama ke baru sesuai permintaan
label_map = {
    1: 0,  # echinus
    2: 1,  # holothurian
    3: 2,  # scallop
    4: 3   # starfish
}

def fix_labels_custom(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)
    new_annotations = []
    for anno in data['annotations']:
        old_cat = anno['category_id']
        if old_cat in label_map:
            anno['category_id'] = label_map[old_cat]
            new_annotations.append(anno)
    data['annotations'] = new_annotations
    # Update categories sesuai mapping baru
    data['categories'] = [
        {"id":0, "name": "echinus"},
        {"id":1, "name": "holothurian"},
        {"id":2, "name": "scallop"},
        {"id":3, "name": "starfish"},
    ]
    with open(output_path, 'w') as f:
        json.dump(data, f)
    print(f"Saved fixed labels to {output_path}")

fix_labels_custom('data/train.json', 'data/train_fixed.json')
fix_labels_custom('data/val.json', 'data/val_fixed.json')
fix_labels_custom('data/test.json', 'data/test_fixed.json')
