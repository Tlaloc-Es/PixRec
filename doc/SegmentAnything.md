# Segment Anything

In this document we will explain how we use this model in the library for experimentation.

## Instalation

All of these spets were given from the original repo that are linked in the references section.

Install Segment Anything:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

or with pipenv

```bash
pipenv install -e git+https://github.com/facebookresearch/segment-anything.git#egg=segment_anything
```

The following optional dependencies are necessary for mask post-processing, saving masks in COCO format, the example notebooks, and exporting the model in ONNX format.

```bash
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

or with pipenv

```bash
pipenv install opencv-python pycocotools matplotlib onnxruntime onnx
```

or just it write

```bash
pipenv install
```

because there are all in the Pipfile ready to install with it

## References

- https://github.com/facebookresearch/segment-anything
