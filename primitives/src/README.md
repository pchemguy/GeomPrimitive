# Overview

This repo contains several components with focus on two objective:  
- Generation of scenes with randomized grids, geometric primitives (lines, elliptic arcs, triangles, and rectangles), randomized styles (color, line patterns, thickness, transparency), handwriting imitation (straight line segments and elliptic arcs are represented by a chain of splines with random deviations from ideal shape; angles and coordinates are jittered). The vector scene is rendered with Matplotlib, rasterized, and a suite of photo like distortions is then introduced as the second stage (noise, rotation, lens distortion, uneven lighting).
- Analysis and enhancement of real photographs of biological samples over millimeter graph paper. The main objective is to detect the background grid, analyze it, and use geometric distortion information for compensation of distortion and subsequent programmatic sample area measurement.
