import argparse
from plantcv import plantcv as pcv
import os
from tqdm import tqdm
import sys
import cv2
import matplotlib.pyplot as plt


def getlastname(path):
    """Extract the last component name from a path."""
    if path.endswith("/"):
        return path.split("/")[-2].split(".")[0]
    return path.split("/")[-1].split(".")[0]


class Transformation:
    def __init__(self, options):
        self.options = options

        # apply write image
        pcv.params.debug_outdir = self.options.outdir
        if self.options.writeimg:
            self.name_save = os.path.join(
                self.options.outdir, getlastname(self.options.image)
            )

        # original
        self.img = None

        # gaussian_blur
        self.blur = None

        # masked
        self.masked2 = None
        self.ab = None

        # roi_objects
        self.roi_objects = None
        self.hierarchy = None
        self.kept_mask = None

        # analysis_objects
        self.mask = None
        self.obj = None

    def original(self):
        img, _, _ = pcv.readimage(filename=self.options.image)
        self.img = img
        return img

    def gaussian_blur(self):
        if self.img is None:
            self.original()

        s = pcv.rgb2gray_hsv(rgb_img=self.img, channel="s")
        s_thresh = pcv.threshold.binary(gray_img=s,
                                        threshold=60, object_type="light")
        s_gblur = pcv.gaussian_blur(img=s_thresh, ksize=(5, 5),
                                    sigma_x=0, sigma_y=None)

        self.blur = s_gblur
        return s_gblur

    def masked(self):
        if self.blur is None:
            self.gaussian_blur()

        b = pcv.rgb2gray_lab(rgb_img=self.img, channel="b")
        b_thresh = pcv.threshold.binary(gray_img=b,
                                        threshold=200, object_type="light")
        bs = pcv.logical_or(bin_img1=self.blur, bin_img2=b_thresh)

        masked = pcv.apply_mask(img=self.img, mask=bs, mask_color="white")

        masked_a = pcv.rgb2gray_lab(rgb_img=masked, channel="a")
        masked_b = pcv.rgb2gray_lab(rgb_img=masked, channel="b")

        maskeda_thresh = pcv.threshold.binary(
            gray_img=masked_a, threshold=115, object_type="dark"
        )
        maskeda_thresh1 = pcv.threshold.binary(
            gray_img=masked_a, threshold=135, object_type="light"
        )
        maskedb_thresh = pcv.threshold.binary(
            gray_img=masked_b, threshold=128, object_type="light"
        )

        ab1 = pcv.logical_or(bin_img1=maskeda_thresh, bin_img2=maskedb_thresh)
        ab = pcv.logical_or(bin_img1=maskeda_thresh1, bin_img2=ab1)
        ab_fill = pcv.fill(bin_img=ab, size=200)

        masked2 = pcv.apply_mask(img=masked, mask=ab_fill, mask_color="white")

        self.masked2 = masked2
        self.ab = ab_fill
        return masked2

    def roi(self):
        if self.masked2 is None:
            self.masked()

        # Find contours using OpenCV
        id_objects, obj_hierarchy = cv2.findContours(
            self.ab, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # For compatibility, we'll use all found objects as ROI objects
        # In modern PlantCV, ROI filtering is done differently
        self.roi_objects = id_objects
        self.hierarchy = (
            obj_hierarchy[0]
            if len(obj_hierarchy) > 0 and obj_hierarchy[0] is not None
            else obj_hierarchy
        )
        self.kept_mask = self.ab  # Use the full mask as kept_mask

        # Calculate total object area
        obj_area = (sum(
            [cv2.contourArea(cnt) for cnt in id_objects]
            ) if id_objects else 0
        )

        return self.roi_objects, self.hierarchy, self.kept_mask, obj_area

    def analysis_objects(self):
        if self.roi_objects is None:
            self.roi()

        # Use the largest contour as the main object
        if len(self.roi_objects) > 0:
            self.obj = max(self.roi_objects, key=cv2.contourArea)
        else:
            self.obj = None

        self.mask = self.kept_mask

        # Draw analysis on image
        analysis_image = self.img.copy()
        if self.obj is not None:
            # Draw contour
            cv2.drawContours(analysis_image, [self.obj], -1, (0, 255, 0), 2)

            # Calculate and draw bounding box
            x, y, w, h = cv2.boundingRect(self.obj)
            cv2.rectangle(analysis_image,
                          (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Calculate properties
            area = cv2.contourArea(self.obj)
            perimeter = cv2.arcLength(self.obj, True)

            # Add text annotations
            cv2.putText(
                analysis_image,
                f"Area: {area:.0f}px",
                (x, y - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                analysis_image,
                f"Perimeter: {perimeter:.0f}px",
                (x, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        return analysis_image

    def pseudolandmarks(self):
        if self.mask is None:
            self.analysis_objects()

        # Create pseudolandmarks visualization manually
        landmark_img = self.img.copy()

        if self.obj is not None and len(self.obj) > 0:
            # Draw center point
            M = cv2.moments(self.obj)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(landmark_img, (cx, cy), 5, (0, 255, 255), -1)

            # Draw equidistant points along contour
            n_landmarks = 100
            perimeter = cv2.arcLength(self.obj, True)
            distance = perimeter / n_landmarks
            current_distance = 0

            cnt = self.obj.reshape(-1, 2)
            for i in range(len(cnt) - 1):
                p1 = cnt[i]
                p2 = cnt[i + 1]
                d = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
                current_distance += d

                if current_distance >= distance:
                    color = (255, 0, 0) if i % 2 == 0 else (0, 0, 255)
                    cv2.circle(landmark_img, tuple(p1), 3, color, -1)
                    current_distance = 0

        # Return dummy values for compatibility
        return None, None, None


class Options:
    def __init__(
        self, path, debug="print",
        writeimg=True,
        result="results.json",
        outdir="."
    ):
        self.image = path
        self.debug = debug
        self.writeimg = writeimg
        self.result = result
        self.outdir = outdir
        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir)


def create_composite_image(transformation, options):
    """Create a composite image showing all 6 transformations in a grid."""
    # Prepare images
    # Convert grayscale to RGB for consistent display
    gaussian_rgb = cv2.cvtColor(transformation.blur, cv2.COLOR_GRAY2RGB)

    # Get ROI image (with contours drawn)
    roi_img = transformation.img.copy()
    if transformation.roi_objects:
        cv2.drawContours(roi_img,
                         transformation.roi_objects, -1, (0, 255, 0), 2)

    # Get analysis image (already created in analysis_objects)
    analysis_img = transformation.img.copy()
    if transformation.obj is not None:
        cv2.drawContours(analysis_img,
                         [transformation.obj], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(transformation.obj)
        cv2.rectangle(analysis_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Get pseudolandmarks image
    landmark_img = transformation.img.copy()
    if transformation.obj is not None and len(transformation.obj) > 0:
        M = cv2.moments(transformation.obj)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(landmark_img, (cx, cy), 5, (0, 255, 255), -1)

        n_landmarks = 100
        perimeter = cv2.arcLength(transformation.obj, True)
        distance = perimeter / n_landmarks
        current_distance = 0

        cnt = transformation.obj.reshape(-1, 2)
        for i in range(len(cnt) - 1):
            p1 = cnt[i]
            p2 = cnt[i + 1]
            d = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
            current_distance += d

            if current_distance >= distance:
                color = (255, 0, 0) if i % 2 == 0 else (0, 0, 255)
                cv2.circle(landmark_img, tuple(p1), 3, color, -1)
                current_distance = 0

    # XOR image (already created in masked method but need to recreate)
    masked_a = pcv.rgb2gray_lab(rgb_img=transformation.masked2, channel="a")
    masked_b = pcv.rgb2gray_lab(rgb_img=transformation.masked2, channel="b")
    maskeda_thresh = pcv.threshold.binary(
        gray_img=masked_a, threshold=115, object_type="dark"
    )
    maskedb_thresh = pcv.threshold.binary(
        gray_img=masked_b, threshold=128, object_type="light"
    )
    xor_img = pcv.logical_xor(bin_img1=maskeda_thresh, bin_img2=maskedb_thresh)
    xor_img_color = pcv.apply_mask(
        img=transformation.img, mask=xor_img, mask_color="white"
    )

    # Create figure with 2 rows and 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"Transformations: {getlastname(options.image)}",
        fontsize=16, fontweight="bold"
    )

    # Convert BGR to RGB for matplotlib display
    images = [
        (transformation.img, "Original"),
        (gaussian_rgb, "Gaussian Blur"),
        (transformation.masked2, "Masked"),
        (xor_img_color, "XOR"),
        (roi_img, "ROI Objects"),
        (analysis_img, "Analysis Objects"),
    ]

    for idx, (img, title) in enumerate(images):
        row = idx // 3
        col = idx % 3
        # Convert BGR to RGB for matplotlib
        if len(img.shape) == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img

        axes[row, col].imshow(img_rgb)
        axes[row, col].set_title(title, fontsize=12, fontweight="bold")
        axes[row, col].axis("off")

    plt.tight_layout()

    # Save composite image
    base = getlastname(options.image)
    # Clean filename (remove special chars that may cause issues)
    clean_base = base.replace(" ", "_").replace("(", "").replace(")", "")
    composite_path = os.path.join(
        options.outdir, clean_base + "_all_transformations.png"
    )
    plt.savefig(composite_path, dpi=150, bbox_inches="tight", format="png")
    plt.close()
    print(f"Saved composite: {composite_path}")


def transform_image(options):
    """Generate all image transformations and create a composite image."""
    transformation = Transformation(options)
    transformation.original()
    transformation.gaussian_blur()
    transformation.masked()
    transformation.roi()
    transformation.analysis_objects()
    transformation.pseudolandmarks()

    # Create composite image with all transformations
    if options.debug == "print":
        create_composite_image(transformation, options)


def recalculate(src, path):
    if not src.endswith("/"):
        src += "/"

    last = getlastname(src)
    relative_path = path[len(src):]

    if relative_path == "":
        return last
    return last + "/" + relative_path


def batch_transform(src, dst):
    if src is None or dst is None:
        raise Exception("Need to specify src and dst")
    if not os.path.isdir(src):
        raise Exception("src is not a dir")
    if not os.path.isdir(dst):
        os.makedirs(dst)

    for root, _, files in os.walk(src):
        name = recalculate(src, root)

        try:
            os.makedirs(os.path.join(dst, name))
        except FileExistsError:
            pass

        print("Doing batch for directory",
              name, "found", len(files), "pictures")
        for file in tqdm(files):
            if file.endswith(".JPG") or file.endswith(".jpg"):
                opt = Options(
                    os.path.join(root, file),
                    debug="print",
                    writeimg=True,
                    outdir=dst + "/" + name,
                )
                try:
                    transform_image(opt)
                except Exception as e:
                    print(f"\nFailed to process {file}: {e}")
        print()


def save_mask_only(src, dst):
    if src is None or dst is None:
        raise Exception("Need to specify src and dst")
    if not os.path.isdir(dst):
        os.makedirs(dst)

    # Single file case
    if os.path.isfile(src):
        opt = Options(src, debug=None, writeimg=False, outdir=dst)
        transformation = Transformation(opt)
        transformation.original()
        transformation.gaussian_blur()
        transformation.masked()

        # Save only the masked image (original with mask applied)
        base = getlastname(src)
        mask_path = os.path.join(dst, base + "_masked.JPG")
        cv2.imwrite(mask_path, transformation.masked2)
        print(f"Saved masked image: {mask_path}")
        return

    # Directory case
    for root, _, files in os.walk(src):
        name = recalculate(src, root)

        try:
            os.makedirs(os.path.join(dst, name))
        except FileExistsError:
            pass

        print("Doing mask batch for directory",
              name, "found", len(files), "pictures")
        for file in tqdm(files):
            if file.endswith(".JPG") or file.endswith(".jpg"):
                opt = Options(
                    os.path.join(root, file),
                    debug=None,
                    writeimg=False,
                    outdir=dst + "/" + name,
                )
                try:
                    transformation = Transformation(opt)
                    transformation.original()
                    transformation.gaussian_blur()
                    transformation.masked()

                    # Save only the masked image (original with mask applied)
                    base = getlastname(opt.image)
                    mask_path = os.path.join(opt.outdir, base + "_masked.JPG")
                    cv2.imwrite(mask_path, transformation.masked2)
                except Exception as e:
                    print(f"\nFailed to process {file}: {e}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-src", type=str,
                        help="Path to the source dir or image.")
    parser.add_argument(
        "-dst",
        type=str,
        default="output/transformations/",
        help="Path to the destination dir. (required for batch processing)",
    )
    parser.add_argument(
        "-mask",
        action="store_true",
        help="Save only binary masks instead of all transformations",
    )
    args = parser.parse_args()

    if not args.src:
        parser.print_help()
        sys.exit(1)

    # Check if src is a file or directory
    if os.path.isfile(args.src):
        # Single file mode
        if not args.dst:
            print("Error: -dst required to save outputs")
            sys.exit(1)

        if args.mask:
            save_mask_only(args.src, args.dst)
        else:
            # Single file all transformations
            opt = Options(args.src,
                          debug="print", writeimg=True, outdir=args.dst)
            transform_image(opt)
    elif os.path.isdir(args.src):
        # Directory mode
        if not args.dst:
            print("Error: -dst required for batch processing")
            sys.exit(1)

        if args.mask:
            save_mask_only(args.src, args.dst)
        else:
            batch_transform(args.src, args.dst)
    else:
        print(f"Error: {args.src} is neither a file nor a directory")
        sys.exit(1)
