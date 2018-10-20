"""
Library for reading multiresolution micro-magellan
"""
import os
import mmap
import numpy as np
import sys
import json

#TIFF constants
WIDTH = 256
HEIGHT = 257
BITS_PER_SAMPLE = 258
COMPRESSION = 259
PHOTOMETRIC_INTERPRETATION = 262
IMAGE_DESCRIPTION = 270
STRIP_OFFSETS = 273
SAMPLES_PER_PIXEL = 277
ROWS_PER_STRIP = 278
STRIP_BYTE_COUNTS = 279
X_RESOLUTION = 282
Y_RESOLUTION = 283
RESOLUTION_UNIT = 296
MM_METADATA = 51123

#file format constants
INDEX_MAP_OFFSET_HEADER = 54773648
INDEX_MAP_HEADER = 3453623
SUMMARY_MD_HEADER = 2355492


def read_header(file):
    # read standard tiff header
    if file[:2] == b'\x4d\x4d':
        # Big endian
        if sys.byteorder != 'big':
            raise Exception("Potential issue with mismatched endian-ness")
    elif file[:2] == b'\x49\x49':
        # little endian
        if sys.byteorder != 'litte':
            raise Exception("Potential issue with mismatched endian-ness")
    else:
        raise Exception('Endian type not specified correctly')
    if np.frombuffer(file[2:4], dtype=np.uint16)[0] != 42:
        raise Exception('Tiff magic 42 missing')
    first_ifd_offset = np.frombuffer(file[4:8], dtype=np.uint32)[0]

    # read custom stuff: summary md, index map
    index_map_offset_header, index_map_offset = np.frombuffer(file[8:16], dtype=np.uint32)
    if index_map_offset_header != INDEX_MAP_OFFSET_HEADER:
        raise Exception('Index map offset header wrong')
    summary_md_header, summary_md_length = np.frombuffer(file[32:40], dtype=np.uint32)
    if summary_md_header != SUMMARY_MD_HEADER:
        raise Exception('Index map offset header wrong')
    summary_md = json.loads(file[40:40 + summary_md_length])
    index_map_header, index_map_length = np.frombuffer(file[40 + summary_md_length:48 + summary_md_length],
                                                       dtype=np.uint32)
    if index_map_header != INDEX_MAP_HEADER:
        raise Exception('Index map header incorrect')
    index_map = [(entry[:4], entry[4]) for entry in
                 np.reshape(np.frombuffer(file[48 + summary_md_length:48 +
                                                                      summary_md_length + index_map_length * 20],
                                          dtype=np.uint32), [-1, 5])]
    return summary_md, index_map, first_ifd_offset

def read_ifd(file, byte_offset):
    num_entries = np.frombuffer(file[byte_offset, byte_offset+2], dtype=np.uint16)[0]
    np.frombuffer(file[byte_offset + 2, num_entries * 12 + 4], dtype=np.uint16)
    #TODO: this is longer if it supports RGB

    # readIntoBuffer(byteOffset + 2, numEntries * 12 + 4)
    #
    # entries = readIntoBuffer(byteOffset + 2, numEntries * 12 + 4).order(byteOrder_);
    #
    # for (int i = 0; i < numEntries; i++) {
    #     IFDEntry entry = readDirectoryEntry(i * 12, entries);
    # if (entry.tag == MM_METADATA) {
    # data.mdOffset = entry.value;
    # data.mdLength = entry.count;
    # } else if (entry.tag == STRIP_OFFSETS) {
    # data.pixelOffset = entry.value;
    # } else if (entry.tag == STRIP_BYTE_COUNTS) {
    # data.bytesPerImage = entry.value;
    # }
    # }
    # data.nextIFD = unsignInt(entries.getInt(numEntries * 12));
    # data.nextIFDOffsetLocation = byteOffset + 2 + numEntries * 12;




dataset_path = '/Users/henrypinkard/Desktop/Magellan data/testdata_1'

res_dirs = [dI for dI in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, dI))]

for res_dir in res_dirs:
    full_path_res_dir = os.path.join(dataset_path, res_dirs[-1])
    tiff_names = [os.path.join(full_path_res_dir, tiff) for tiff in os.listdir(full_path_res_dir)]
    for tiff in tiff_names:
        with open(tiff, 'r+b') as file:
            # memory map the entire file
            mm = mmap.mmap(file.fileno(), 0)
            summary_md, index_map, first_ifd_offset = read_header(mm)
            

            mm.close()
            pass