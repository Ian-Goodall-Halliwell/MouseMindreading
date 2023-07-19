
from pathlib import Path
import os

from allensdk.core.reference_space_cache import ReferenceSpaceCache

output_dir = '.'
reference_space_key = os.path.join('annotation', 'ccf_2017')
resolution = 25
rspc = ReferenceSpaceCache(resolution, reference_space_key, manifest=Path(output_dir) / 'manifest.json')
# ID 1 is the adult mouse structure graph
tree = rspc.get_structure_tree(structure_graph_id=1) 
# now let's take a look at a structure
tree.get_structures_by_name(['Dorsal auditory area'])
[{'acronym': 'AUDd',
  'graph_id': 1,
  'graph_order': 122,
  'id': 1011,
  'name': 'Dorsal auditory area',
  'structure_id_path': [997, 8, 567, 688, 695, 315, 247, 1011],
  'structure_set_ids': [112905828,
   688152357,
   691663206,
   687527945,
   12,
   184527634,
   167587189,
   114512891],
  'rgb_triplet': [1, 147, 153]}]

# get a structure's parent
tree.parents([1011])
[{'acronym': 'AUD',
  'graph_id': 1,
  'graph_order': 121,
  'id': 247,
  'name': 'Auditory areas',
  'structure_id_path': [997, 8, 567, 688, 695, 315, 247],
  'structure_set_ids': [3, 112905828, 691663206, 12, 184527634, 114512891],
  'rgb_triplet': [1, 147, 153]}]
# get a dictionary mapping structure ids to names

name_map = tree.get_name_map()
name_map[247]
'Auditory areas'
# ask whether one structure is contained within another

structure_id_a = 385
structure_id_b = 247

is_desc = '' if tree.structure_descends_from(structure_id_a, structure_id_b) else ' not'
print( '{0} is{1} in {2}'.format(name_map[structure_id_a], is_desc, name_map[structure_id_b]) )

# build a custom map that looks up acronyms by ids
# the syntax here is just a pair of node-wise functions. 
# The first one returns keys while the second one returns values

acronym_map = tree.value_map(lambda x: x['id'], lambda y: y['acronym'])
print( acronym_map[structure_id_a] )

import os

annotation, meta = rspc.get_annotation_volume()
# The file should appear in the reference space key directory
os.listdir(Path(output_dir) / reference_space_key)

rsp = rspc.get_reference_space()
import matplotlib.pyplot as plt


# A complete mask for one structure
whole_cortex_mask = rsp.make_structure_mask([315])

# view in coronal section
fig, ax = plt.subplots(figsize=(10, 10))
plt.imshow(whole_cortex_mask[150, :], interpolation='none', cmap=plt.cm.afmhot)


# This gets all of the structures targeted by the Allen Brain Observatory project
brain_observatory_structures = rsp.structure_tree.get_structures_by_set_id([514166994])
brain_observatory_ids = [st['id'] for st in brain_observatory_structures]

brain_observatory_mask = rsp.make_structure_mask(brain_observatory_ids)

# view in horizontal section
fig, ax = plt.subplots(figsize=(10, 10))
plt.imshow(brain_observatory_mask[:, 40, :], interpolation='none', cmap=plt.cm.afmhot)

import functools
from allensdk.core.reference_space import ReferenceSpace

# Define a wrapper function that will control the mask generation. 
# This one checks for a nrrd file in the specified base directory 
# and builds/writes the mask only if one does not exist
annotation_dir = Path(output_dir) / 'annotation'
mask_writer = functools.partial(ReferenceSpace.check_and_write, annotation_dir)
    
# many_structure_masks is a generator - nothing has actrually been run yet
mask_generator = rsp.many_structure_masks([385, 1097], mask_writer)

# consume the resulting iterator to make and write the masks
for structure_id in mask_generator:
    print( 'made mask for structure {0}.'.format(structure_id) ) 

os.listdir(annotation_dir)

# Double-check the voxel counts
no_voxel_id = rsp.structure_tree.get_structures_by_name(['Somatosensory areas, layer 6a'])[0]['id']
print( 'voxel count for structure {0}: {1}'.format(no_voxel_id, rsp.total_voxel_map[no_voxel_id]) )

# remove unassigned structures from the ReferenceSpace's StructureTree
rsp.remove_unassigned()

# check the structure tree
no_voxel_id in rsp.structure_tree.node_ids()

import numpy as np

fig, ax = plt.subplots(figsize=(10, 10))
plt.imshow(rsp.get_slice_image(1, 5000), interpolation='none')

import warnings

target_resolution = [75, 75, 75]

# in some versions of scipy, scipy.ndimage.zoom raises a helpful but distracting 
# warning about the method used to truncate integers. 
warnings.simplefilter('ignore')

sf_rsp = rsp.downsample(target_resolution)

# re-enable warnings
warnings.simplefilter('default')

print( rsp.annotation.shape )
print( sf_rsp.annotation.shape )


fig, ax = plt.subplots(figsize=(10, 10))
plt.imshow(sf_rsp.get_slice_image(1, 5000), interpolation='none')

# using the downsampled annotations
hm_rsp = rsp.downsample([100, 100, 100])
hm_rsp.write_itksnap_labels('ccf_2017_itksnap.nrrd', 'ccf_2017_itksnap_labels.txt')