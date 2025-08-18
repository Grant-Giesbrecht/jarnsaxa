import textwrap
from itertools import groupby
import re
import h5py
import time
import json
import numpy as np
from abc import ABC, abstractmethod
import pylogfile as plf
from colorama import Fore, Style
import copy

class Packable(ABC):
	""" This class represents all objects that can be packed and unpacked and sent between the client and server
	
	manifest, obj_manifest, and list_template are all dictionaries. Each describes a portion of how to represent a class as a string,
	and how to convert back to the class from the string data.
	
	manifest: lists all variables that can be converted to/from JSON natively
	obj_manifest: lists all variables that are Packable objects or lists of packable objects. Each object will have
		pack() called, and be understood through its unpack() function.
	list_template: dictionary mapping item to pack/unpack to its class type, that way during unpack, Packable knows which
		class to create and call unpack() on.
	
	Populate all three of these variables as needed in the set_manifest function. set_manifest is called in super().__init__(), so
	it shouldn't need to be remembered in any of the child classes.
	"""
	
	def __init__(self, log:plf.LogPile=None):
		
		if log is None:
			print(f"{Fore.RED}Log object was not initialized in {type(self)} instance.!{Style.RESET_ALL}")
			self.log = plf.LogPile(use_mutex=False)
		else:
			self.log = log
			self.log.set_enable_mutex(False)
		
		self.log.lowdebug(f"Created object type={type(self)}")
		self.manifest = []
		self.obj_manifest = []
		self.list_manifest = {}
		self.dict_manifest = {}
		
		self.set_manifest()
	
	@abstractmethod
	def set_manifest(self):
		""" This function will populate the manifest and obj_manifest objects"""
		pass
	
	def pack(self):
		""" Returns the object to as a JSON dictionary """
		
		# Initialize dictionary
		d = {}
		
		# Add items in manifest to packaged data
		for mi in self.manifest:
			d[mi] = getattr(self, mi)
		
		# Scan over object manifest
		for mi in self.obj_manifest:
			# Pack object and add to output data
			try:
				d[mi] = getattr(self, mi).pack()
			except:
				raise Exception(f"'Packable' object had corrupt object manifest item '{mi}'. Cannot pack.", detail=f"Type = {type(self)}")
				
		# Scan over list manifest
		for mi in self.list_manifest:
				
			# Pack objects in list and add to output data
			d[mi] = [x.pack() for x in getattr(self, mi)]
		
		# Scan over dict manifest
		for mi in self.dict_manifest:
			
			mi_deref = getattr(self, mi)
			
			# Pack objects in dict and add to output data
			d[mi] = {}
			for midk in mi_deref.keys():
				d[mi][midk] = mi_deref[midk].pack()
				
		# Return data list
		return d
	
	def unpack(self, data:dict):
		""" Populates the object from a JSON dict """
		
		# Try to populate each item in manifest
		for mi in self.manifest:
			self.log.lowdebug(f"Unpacking manifest, item:>{mi}<")
			
			# Try to assign the new value
			try:
				setattr(self, mi, data[mi])
			except Exception as e:
				self.log.error(f"Failed to unpack item in object of type '{type(self).__name__}'. ({e})", detail=f"Type = {type(self)}")
				return
		
		# Try to populate each Packable object in manifest
		for mi in self.obj_manifest:
			self.log.lowdebug(f"Unpacking obj_manifest, item:>{mi}<")
			
			# Try to update the object by unpacking the item
			try:
				getattr(self, mi).unpack(data[mi])
			except Exception as e:
				self.log.error(f"Failed to unpack Packable in object of type '{type(self).__name__}'. ({e})", detail=f"Type = {type(self)}")
				return
			
		# Try to populate each list of Packable objects in manifest
		for mi in self.list_manifest.keys():
				
			# Scan over list, unpacking each element
			temp_list = []
			for list_item in data[mi]:
				self.log.lowdebug(f"Unpacking list_manifest, item:>{mi}<, element:>:a{list_item}<")
				
				# Try to create a new object and unpack a list element
				try:
					# Create a new object of the correct type
					new_obj = copy.deepcopy(self.list_manifest[mi])
					
					# Populate the new object by unpacking it, add to list
					new_obj.unpack(list_item)
					temp_list.append(new_obj)
				except Exception as e:
					self.log.error(f"Failed to unpack list of Packables in object of type '{type(self).__name__}'. Type={type(self)}. ({e})", detail=f"Type = {type(self)}")
					return
			setattr(self, mi, temp_list)
				# self.obj_manifest[mi] = copy.deepcopy(temp_list)
		
		# Scan over dict manifest
		for mi in self.dict_manifest.keys():
			
			# mi_deref = getattr(self, mi)
			
			# # Pack objects in list and add to output data
			# d[mi] = [mi_deref[midk].pack() for midk in mi_deref.keys()]
			
			# Scan over list, unpacking each element
			temp_dict = {}
			for dmk in data[mi].keys():
				self.log.lowdebug(f"Unpacking manifest, item:>{mi}<, element:>:a{dmk}<")
				
				# Try to create a new object and unpack a list element
				try:
					# Create a new object of the correct type
					try:
						new_obj = copy.deepcopy(self.dict_manifest[mi])
					except Exception as e:
						print("Dict Manifest:")
						print(self.dict_manifest)
						self.log.error(f"Failed to unpack dict_manifest[{mi}], ({e})")
						return
					
					# Populate the new object by unpacking it, add to list
					new_obj.unpack(data[mi][dmk])
					temp_dict[dmk] = new_obj
				except Exception as e:
					prob_item = data[mi][dmk]
					self.log.error(f"Failed to unpack dict of Packables in object of type '{type(self).__name__}'. ({e})", detail=f"Class={type(self)}, problem manifest item=(name:{dmk}, type:{type(prob_item)})")
					return
			setattr(self, mi, temp_dict)

def wrap_text(text:str, width:int=80):
	""" Accepts a string, and wraps it over multiple lines. Honors line breaks. Returns a single string."""
	
	# Break at \n and form list of strings without \n
	split_lines = text.splitlines()
	
	all_lines = []
	
	# Loop over each split string, apply standard wrap
	for sl in split_lines:
		
		wt = textwrap.wrap(sl, width=width)
		for wtl in wt:
			all_lines.append(wtl)
	
	# Join with newline characters
	return '\n'.join(all_lines)

def ensureWhitespace(s:str, targets:str, whitespace_list:str=" \t", pad_char=" "):
	""" """
	
	# Remove duplicate targets
	targets = "".join(set(targets))
	
	# Add whitespace around each target
	for tc in targets:
		
		start_index = 0
		
		# Find all instances of target
		while True:
			
			# Find next instance of target
			try:
				idx = s[start_index:].index(tc)
				idx += start_index
			except ValueError as e:
				break # Break when no more instances
			
			# Update start index
			start_index = idx + 1
			
			# Check if need to pad before target
			add0 = True
			if idx == 0:
				add0 = False
			elif s[idx-1] in whitespace_list:
				add0 = False
			
			# Check if need to pad after target
			addf = True
			if idx >= len(s)-1:
				addf = False
			elif s[idx+1] in whitespace_list:
				addf = False
			
			# Add required pad characters
			if addf:
				s = s[:idx+1] + pad_char + s[idx+1:]
				start_index += 1 # Don't scan pad characters
			if add0:
				s = s[:idx] + pad_char + s[idx:]
	
	return s

def barstr(text:str, width:int=80, bc:str='*', pad:bool=True):

		s = text;

		# Pad input if requested
		if pad:
			s = " " + s + " ";

		pad_back = False;
		while len(s) < width:
			if pad_back:
				s = s + bc
			else:
				s = bc + s
			pad_back = not pad_back

		return s

class StringIdx():
	def __init__(self, val:str, idx:int, idx_end:int=-1):
		self.str = val
		self.idx = idx
		self.idx_end = idx_end

	def __str__(self):
		return f"[{self.idx}]\"{self.str}\""

	def __repr__(self):
		return self.__str__()

def parse_idx(input:str, delims:str=" ", keep_delims:str=""):
	""" Parses a string, breaking it up into an array of words. Separates at delims. """
	
	def parse_two_idx(input:str, delims:str):
		p = 0
		for k, g in groupby(input, lambda x:x in delims):
			q = p + sum(1 for i in g)
			if not k:
				yield (p, q) # or p, q-1 if you are really sure you want that
			p = q
	
	out = []
	
	sections = list(parse_two_idx(input, delims))
	for s in sections:
		out.append(StringIdx(input[s[0]:s[1]], s[0], s[1]))
	return out

def dict_to_hdf(root_data:dict, save_file:str, use_json_backup:bool=False, show_detail:bool=False) -> bool:
	''' Writes a dictionary to an HDF file per the rules used by 'write_level()'. 
	
	* If the value of a key in another dictionary, the key is made a group (directory).
	* If the value of the key is anything other than a dictionary, it assumes
	  it can be saved to HDF (such as a list of floats), and saves it as a dataset (variable).
	
	Args:
		root_data (dict): Dictionary to save to file.
		save_file (str): Filename to write to.
		use_json_backup (bool): Optional parameter to save a copy of the file as a JSON dict. Default = False.
		show_detail (bool): Optional parameter to show detail while saving. Default = False.
	
	Returns:
		bool: True if successfully saved.
	'''
	
	def write_level(fh:h5py.File, level_data:dict, show_detail:bool=False):
		''' Writes a dictionary to the hdf file.
		
		Recursive function used by  '''
		
		# Scan over each directory of root-data
		for k, v in level_data.items():
			
			# If value is a dictionary, this key represents a directory
			if type(v) == dict:
				
				# Create a new group
				fh.create_group(k)
				
				# Write the dictionary to the group
				write_level(fh[k], v)
					
			else: # Otherwise try to write this datatype (ex. list of floats)
				
				# Write value as a dataset
				try:
					fh.create_dataset(k, data=v)
				except Exception as e:
					if show_detail:
						print(f"Failed to write dataset '{k}' with value of type {type(v)}. ({e})")
					return False
		return True
	
	# Start timer
	t0 = time.time()
	
	# Open HDF
	hdf_successful = True
	exception_str = ""
	
	# Recursively write HDF file
	with h5py.File(save_file, 'w') as fh:
		
		# Try to write dictionary
		try:
			if not write_level(fh, root_data, show_detail=show_detail):
				hdf_successful = False
				exception_str = "Set show_detail to true for details"
		except Exception as e:
			hdf_successful = False
			exception_str = f"{e}"
	
	# Check success condition
	if hdf_successful:
		# print(f"Wrote file in {time.time()-t0} sec.")
		
		return True
	else:
		print(f"Failed to write HDF file! ({exception_str})")
		
		# Write JSON as a backup if requested
		if use_json_backup:
			
			# Add JSON extension
			save_file_json = save_file[:-3]+".json"
			
			# Open and save JSON file
			try:
				with open(save_file_json, "w") as outfile:
					outfile.write(json.dumps(root_data, indent=4))
			except Exception as e:
				print(f"Failed to write JSON backup: ({e}).")
				return False
		
		return True

def hdf_to_dict(filename, to_lists:bool=True, decode_strs:bool=True) -> dict:
	''' Reads a HDF file and converts the data to a dictionary '''
	
	def read_level(fh:h5py.File) -> dict:
		
		# Initialize output dict
		out_data = {}
		
		# Scan over each element on this level
		for k in fh.keys():
			
			# Read value
			if type(fh[k]) == h5py._hl.group.Group: # If group, recusively call
				out_data[k] = read_level(fh[k])
			else: # Else, read value from file
				out_data[k] = fh[k][()]
				
				# Converting to a pandas DataFrame will crash with
				# some numpy arrays, so convert to a list.
				is_list_type = False
				if type(out_data[k]) == np.ndarray and to_lists:
						out_data[k] = list(out_data[k])
						is_list_type = True
				elif type(out_data[k]) == list and not to_lists:
						out_data[k] = np.array(out_data[k])
						is_list_type = True
				
				if is_list_type and len(out_data[k]) > 0 and type(out_data[k][0]) == bytes:
					for idx, val in enumerate(out_data[k]):
						out_data[k][idx] = val.decode()
				else:
					if decode_strs and type(out_data[k]) == bytes:
						out_data[k] = out_data[k].decode()
				
		return out_data
	
	# Open file
	with h5py.File(filename, 'r') as fh:
		
		try:
			root_data = read_level(fh)
		except Exception as e:
			print(f"Failed to read HDF file! ({e})")
			return None
	
	# Return result
	return root_data