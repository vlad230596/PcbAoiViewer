import cv2

from stitching import Stitcher
from stitching import AffineStitcher
from stitching.images import Images
from stitching.stitching_error import StitchingError
class PlaneStitcher:

    class Result:
        def __init__(self, status, imgs=None, masks=None, corners=None, sizes=None, panorama=None):
            self.status = status
            self.imgs = imgs
            self.masks = masks
            self.corners = corners
            self.sizes = sizes
            self.panorama = panorama
            pass
    def __init__(self):
        pass

    @staticmethod
    def stitch(images, feature_masks=[]):
        # stitcher = AffineStitcher(confidence_threshold=0.5, crop=False, nfeatures=10000, blender_type="no", finder="no", compensator="no")
        stitcher = AffineStitcher(confidence_threshold=0.5, crop=False, nfeatures=10000)

        try:

            # return stitcher.stitch_verbose(images, feature_masks, f'verbose')
            images = Images.of(
                images, stitcher.medium_megapix, stitcher.low_megapix, stitcher.final_megapix
            )

            # Resize Images
            imgs = list(images.resize(Images.Resolution.MEDIUM))

            # Find Features
            finder = stitcher.detector
            features = stitcher.find_features(imgs, feature_masks)

            # Match Features
            matcher = stitcher.matcher
            matches = matcher.match_features(features)

            # Subset
            subsetter = stitcher.subsetter

            all_relevant_matches = list(
                matcher.draw_matches_matrix(
                    imgs,
                    features,
                    matches,
                    conf_thresh=subsetter.confidence_threshold,
                    inliers=True,
                    matchColor=(0, 255, 0),
                )
            )
            # for idx1, idx2, img in all_relevant_matches:
            #     cv2.imshow('featureMatch', img)

            # Subset
            subsetter = stitcher.subsetter
            indices = subsetter.get_indices_to_keep(features, matches)

            imgs = subsetter.subset_list(imgs, indices)
            features = subsetter.subset_list(features, indices)
            matches = subsetter.subset_matches(matches, indices)
            images.subset(indices)

            # Camera Estimation, Adjustion and Correction
            camera_estimator = stitcher.camera_estimator
            camera_adjuster = stitcher.camera_adjuster
            wave_corrector = stitcher.wave_corrector

            cameras = camera_estimator.estimate(features, matches)
            cameras = camera_adjuster.adjust(features, matches, cameras)
            cameras = wave_corrector.correct(cameras)

            # Warp Images
            low_imgs = list(images.resize(Images.Resolution.LOW, imgs))
            imgs = None  # free memory

            warper = stitcher.warper
            warper.set_scale(cameras)

            low_sizes = images.get_scaled_img_sizes(Images.Resolution.LOW)
            camera_aspect = images.get_ratio(Images.Resolution.MEDIUM, Images.Resolution.LOW)

            low_imgs = list(warper.warp_images(low_imgs, cameras, camera_aspect))
            low_masks = list(warper.create_and_warp_masks(low_sizes, cameras, camera_aspect))
            low_corners, low_sizes = warper.warp_rois(low_sizes, cameras, camera_aspect)

            final_sizes = images.get_scaled_img_sizes(Images.Resolution.FINAL)
            camera_aspect = images.get_ratio(Images.Resolution.MEDIUM, Images.Resolution.FINAL)

            final_imgs = list(images.resize(Images.Resolution.FINAL))
            final_imgs = list(warper.warp_images(final_imgs, cameras, camera_aspect))
            final_masks = list(
                warper.create_and_warp_masks(final_sizes, cameras, camera_aspect)
            )
            final_corners, final_sizes = warper.warp_rois(final_sizes, cameras, camera_aspect)

            # Crop
            cropper = stitcher.cropper

            if cropper.do_crop:
                mask = cropper.estimate_panorama_mask(
                    low_imgs, low_masks, low_corners, low_sizes
                )

                low_corners = cropper.get_zero_center_corners(low_corners)
                cropper.prepare(low_imgs, low_masks, low_corners, low_sizes)

                low_masks = list(cropper.crop_images(low_masks))
                low_imgs = list(cropper.crop_images(low_imgs))
                low_corners, low_sizes = cropper.crop_rois(low_corners, low_sizes)

                lir_aspect = images.get_ratio(Images.Resolution.LOW, Images.Resolution.FINAL)
                final_masks = list(cropper.crop_images(final_masks, lir_aspect))
                final_imgs = list(cropper.crop_images(final_imgs, lir_aspect))
                final_corners, final_sizes = cropper.crop_rois(
                    final_corners, final_sizes, lir_aspect
                )

            # Seam Masks
            seam_finder = stitcher.seam_finder

            seam_masks = seam_finder.find(low_imgs, low_corners, low_masks)
            seam_masks = [
                seam_finder.resize(seam_mask, mask)
                for seam_mask, mask in zip(seam_masks, final_masks)
            ]

            # Exposure Error Compensation
            compensator = stitcher.compensator

            compensator.feed(low_corners, low_imgs, low_masks)

            compensated_imgs = [
                compensator.apply(idx, corner, img, mask)
                for idx, (img, mask, corner) in enumerate(
                    zip(final_imgs, final_masks, final_corners)
                )
            ]

            # Blending
            blender = stitcher.blender
            blender.prepare(final_corners, final_sizes)
            for img, mask, corner in zip(compensated_imgs, seam_masks, final_corners):
                blender.feed(img, mask, corner)
            panorama, _ = blender.blend()

            # blended_seam_masks = seam_finder.blend_seam_masks(
            #     seam_masks, final_corners, final_sizes
            # )
            #
            # with_seam_polygons = seam_finder.draw_seam_polygons(panorama, blended_seam_masks)
            # cv2.imshow('with_seam_polygons', with_seam_polygons)
            return PlaneStitcher.Result(True, final_imgs, seam_masks, final_corners, final_sizes, panorama)
        except StitchingError:
            return PlaneStitcher.Result(False)