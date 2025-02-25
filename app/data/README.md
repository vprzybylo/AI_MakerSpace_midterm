# Data Directory

This directory contains the Grid Code documentation and processed data.

## Structure

- `raw/` - Contains the original Grid Code PDF
- `processed/` - Contains processed chunks and embeddings
- `test/` - Contains test data and evaluation sets

## Grid Code PDF

Place the Grid Code PDF file in the `raw/` directory with filename `grid_code.pdf`.

## Processing

The data processing pipeline:
1. Loads PDF from raw/
2. Splits into chunks
3. Generates embeddings
4. Stores processed data

## Test Data

The test directory contains:
- Sample questions and answers
- Evaluation datasets
- Test PDF segments 