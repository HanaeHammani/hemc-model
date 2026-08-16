"""HEMC: Hierarchical Eye Movement Classification.

Stage 1 reproduces and extends EMCCF (Wang et al., 2024, IEEE JBHI) to classify
raw gaze samples into fixation / saccade / smooth pursuit / blink, then applies a
CRF-Viterbi sequential decoder to remove temporal over-segmentation.

Stage 2 takes the fixation samples identified in Stage 1 and further splits them
into microsaccade / drift using hard-negative mining and a ResNet-TS classifier.
"""

__version__ = "0.1.0"
