#!/usr/bin/env python3
"""Run tests from this directory as follows: python3 -m unittest -v test"""

import unittest

import numpy as np
from vmodel.geometry import is_angle_in_interval, tangent_points_to_circle
from vmodel.visibility import visibility_set, is_occluding


class Test(unittest.TestCase):

    def test_tangent_points_to_circle_quadrant_1(self):

        center = np.array([3., 2.])
        radius = 0.5
        pt1_true = np.array([3.2169780165, 1.5495329753])
        pt2_true = np.array([2.6676373681, 2.3735439478])

        pt1, pt2 = tangent_points_to_circle(center, radius)

        self.assertTrue(np.allclose(pt1, pt1_true))
        self.assertTrue(np.allclose(pt2, pt2_true))

    def test_tangent_points_to_circle_quadrant_2(self):

        center = np.array([-4., 2.])
        radius = 1.
        pt1_true = np.array([-3.3641101056, 2.7717797887])
        pt2_true = np.array([-4.2358898944, 1.0282202113])

        pt1, pt2 = tangent_points_to_circle(center, radius)

        self.assertTrue(np.allclose(pt1, pt1_true))
        self.assertTrue(np.allclose(pt2, pt2_true))

    def test_tangent_points_to_circle_angles(self):

        center = np.array([6., 0.])
        radius = 1.
        angle1_true, angle2_true = -9.5940682269, 9.5940682269

        pt1, pt2 = tangent_points_to_circle(center, radius)

        angle1 = np.rad2deg(np.arctan2(pt1[1], pt1[0]))
        angle2 = np.rad2deg(np.arctan2(pt2[1], pt2[0]))

        self.assertAlmostEqual(angle1, angle1_true)
        self.assertAlmostEqual(angle2, angle2_true)

    def test_tangent_points_to_circle_within_circle(self):
        center = np.array([1., 1.])
        radius = 2.
        self.assertIsNone(tangent_points_to_circle(center, radius))

    def test_tangent_points_to_circle_on_circle(self):
        center = np.array([1., 0.])
        radius = 1.
        pt1_true = pt2_true = np.array([0., 0.])
        pt1, pt2 = tangent_points_to_circle(center, radius)
        self.assertTrue(np.allclose(pt1, pt1_true))
        self.assertTrue(np.allclose(pt2, pt2_true))

    def test_is_occluding_full_occlusion(self):
        center1, center2 = np.array([6., 2.]), np.array([9., 3.])
        radius1, radius2 = 1., 1.
        self.assertTrue(is_occluding(center1, radius1, center2, radius2))

    def test_is_occluding_distance_larger(self):
        center1, center2 = np.array([6., 5.]), np.array([6., 2.])
        radius1, radius2 = 1., 1.
        self.assertFalse(is_occluding(center1, radius1, center2, radius2))

    def test_is_occluding_distance_larger_radius_closer(self):
        center1, center2 = np.array([6., 5.]), np.array([6., 2.])
        radius1, radius2 = 4., 1.
        self.assertTrue(is_occluding(center1, radius1, center2, radius2))

    def test_is_occluding_no_occlusion(self):
        center1, center2 = np.array([2., 3.]), np.array([6., 2.])
        radius1, radius2 = 1., 1.
        self.assertFalse(is_occluding(center1, radius1, center2, radius2))

    def test_is_occluding_no_occlusion_quadrants(self):
        center1, center2 = np.array([6., -2.]), np.array([6., 2.])
        radius1, radius2 = 1., 1.
        self.assertFalse(is_occluding(center1, radius1, center2, radius2))

    def test_is_occluding_same_circle(self):
        center1, center2 = np.array([6., 0.]), np.array([6., 0.])
        radius1, radius2 = 1., 1.
        self.assertTrue(is_occluding(center1, radius1, center2, radius2))

    def test_is_occluding_smaller_within(self):
        center1, center2 = np.array([6., 0.]), np.array([6., 0.])
        radius1, radius2 = 0.5, 1.
        self.assertFalse(is_occluding(center1, radius1, center2, radius2))

    def test_is_occluding_smaller_point_within_both_circles(self):
        center1, center2 = np.array([1., 0.]), np.array([2., 0.])
        radius1, radius2 = 2., 3.
        self.assertTrue(is_occluding(center1, radius1, center2, radius2))

    def test_is_occluding_further_away_but_larger_quadrant(self):
        center1, center2 = np.array([7., -1.]), np.array([6., 2.])
        radius1, radius2 = 4., 1.
        self.assertTrue(is_occluding(center1, radius1, center2, radius2))

    def test_is_angle_in_interval_inside_normal(self):
        angle = np.deg2rad(45)
        interval = np.array((np.deg2rad(0), np.deg2rad(90)))
        self.assertTrue(is_angle_in_interval(angle, interval))

    def test_is_angle_in_interval_inside_quadrant(self):
        angle = np.deg2rad(300)
        interval = np.array((np.deg2rad(270), np.deg2rad(45)))
        self.assertTrue(is_angle_in_interval(angle, interval))

    def test_is_angle_in_interval_outside_normal(self):
        angle = np.deg2rad(45)
        interval = np.array((np.deg2rad(90), np.deg2rad(180)))
        self.assertFalse(is_angle_in_interval(angle, interval))

    def test_is_angle_in_interval_outside_quadrant(self):
        angle = np.deg2rad(45)
        interval = np.array((np.deg2rad(340), np.deg2rad(25)))
        self.assertFalse(is_angle_in_interval(angle, interval))

    def test_compute_occlusions_random1(self):
        radius = 0.5
        occluded_agents_true = np.array([3, 5, 7])  # 1-indexed!
        positions = np.array([[2.83778229, 3.0783252],
                              [-1.46653437, 1.80453313],
                              [-2.80929834, 1.67320948],
                              [-2.38409982, 0.6235275],
                              [-2.91558909, 0.22018174],
                              [2.37974831, -3.56279092],
                              [-1.98530291, 1.58483259],
                              [3.81785359, 0.99041497],
                              [0.12622196, -1.98584117]])
        occluded = ~visibility_set(positions, radius)
        occluded_agents = np.array([i + 1 for i in range(len(positions)) if occluded[i]])
        self.assertTrue(np.all(occluded_agents == occluded_agents_true))

    def test_compute_occlusions_random2(self):
        radius = 0.5
        occluded_agents_true = np.array([2, 5, 6])  # 1-indexed!
        positions = np.array([[-2.75114029, -2.06400953],
                              [-2.04081727, 2.16765629],
                              [3.66583955, -1.15494228],
                              [2.49819317, -3.82918581],
                              [-2.99967231, 3.72956997],
                              [-1.82957993, 3.58261861],
                              [-3.85311192, 1.22281318],
                              [2.09105496, 3.36389395],
                              [-1.04757504, 1.94455198]])
        occluded = ~visibility_set(positions, radius)
        occluded_agents = np.array([i + 1 for i in range(len(positions)) if occluded[i]])
        self.assertTrue(np.all(occluded_agents == occluded_agents_true))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
