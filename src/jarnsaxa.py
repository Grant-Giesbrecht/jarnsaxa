from __future__ import annotations
import textwrap
from itertools import groupby
import re
import h5py
import time
import json
import numpy as np
from abc import ABC, abstractmethod
import pylogfile.base as plf
from colorama import Fore, Style
import copy
from dataclasses import dataclass
from typing import Any, Type, Optional, Any, Sequence #Callable, Dict,
from datetime import datetime, timezone
import numpy as np



SERIALIZER_FORMAT_VERSION = 1  # bump when your Serializer file shape/semantics change

@dataclass
class ClassRegistryInfo:
	''' Class used to describe Serializable classes for the class registry. This
	indicates the appropriate 'to' and 'from' functions for creating and
	serializing each class. Additionally, it ensures only registered classes can
	be de-encoded, preventing arbitrary code execution.
	'''
	
	cls: Type # Class type
	to: callable[[Any], dict] # Function for converting to serial
	from_: callable[[dict], Any] # Function for converting from serial
	version: int # Version of class state data
	
	# Optional function to upgrade class data to a newer version
	upgrade: Optional[callable[[dict, int, int], dict]] = None

# A registry so only known classes can be reconstructed
SERIALIZABLE_CLASS_REGISTRY: dict[str, ClassRegistryInfo] = {}

def to_serial_dict(obj:Any) -> dict:
	""" Converts an object to a state specifying serialized dictionary.
	
	Args:
		obj: Object to serialize. Must inherit `Serializable` class.
	
	Returns:
		dict: Serialized object.
	"""
	format_dict = {"name": "ganymede.Serializable", "version": SERIALIZER_FORMAT_VERSION}
	return {"__serializer_format__": format_dict, "state": Serializable.serialize(obj)}

def from_serial_dict(serial_data:dict) -> Any:
	""" Converts serialized-object data in dictionary format and returns a new
	object of the specified type.
	
	Args:
		serial_data (dict): Dictionary describing object state.
	
	Returns:
		New object
	"""
	return Serializable.deserialize(serial_data['state'])

def restore_state(filename:str):
	""" Returns an object that was serialized to dictionary format and reads
	it from disk."""
	
	#TODO: Make it support HDF and JSON, with flags and autodetect
	# Load JSON file
	with open(filename, "r", encoding="utf-8") as f:
		json_data = json.load(f)
	
	# Turn into an object
	return from_serial_dict(json_data)

def dump_state(obj, filename:str) -> None:
	""" Serializes the provided object and dumps it to disk.
	
	Args:
		obj: Serializable object to save. Must inherit `Serializable` class.
		filename (str): File to save.
	
	Returns:
		None
	"""
	
	# Create serialized dict from object
	serial_dict = to_serial_dict(obj)
	
	#TODO: Make it support HDF and JSON, with flags and autodetect
	# Save to file
	with open(filename, "w", encoding="utf-8") as f:
		json.dump(serial_dict, f, ensure_ascii=False, indent=2)


class Serializable:
	''' Class that allows the class data to be serialized to a dictionary format
	for storage or transfer. Works by requiring any class data that needs to be
	preserved to be listen in the __state_fields__ parameter.
	
	Making a subclass of Serializable:
		- You can create a function called __post_deserialize__ which will be called
		  like an __init__ function, but after the object is created from serial
		  data.
		- Populate __state_fields__ with the names of every variable you want saved
		  or loaded to/from the serialized versions.
	
	'''
	
	__state_fields__:Sequence[str] = ()
	__schema_version__:int = 1
	__auto_register__:bool = True  # opt-out switch for ABCs etc.

	def get_state_dict(self) -> dict[str, Any]:
		''' Creates a dictionary with all registered state fields. The data in this
		dictionary is not serialized/packed, but will still contain objects.
		'''
		return {param: getattr(self, param) for param in self.__state_fields__}

	@classmethod
	def from_state_dict(cls, data:dict[str, Any]):
		''' Creates a new object of this class type, and populates the parameters
		listed in __state_fields__ with the values in `data`. 
		
		Parameters:
			data (dict): Dictionary from which to repopulate the object. This
				dictioanry should already be decoded and should have any objects
				in object form, not in serialized form.
		
		Returns:
			New instance of the class, with popualted state parameters.
		'''
		
		# Create a  new instance of the class
		obj = cls.__new__(cls)  # no __init__ call
		
		# Scan over all data items and populate class parameters
		for k, v in data.items():
			setattr(obj, k, v)
		
		# If __post_deserialzie__ is populated for this class type, call it.
		post = getattr(obj, "__post_deserialize__", None)
		if callable(post):
			post()
		
		# Return the finished object
		return obj
	
	@classmethod
	def _register_json_class(cls:Type, *, version:int|None=None, to:callable[[Any], dict]=None, from_:callable[[dict], Any]=None, upgrade:callable[[dict, int, int], dict]=None):
		''' Registers a class in the class registry global variable. All neccesary data
		is automatically infered by just passing the class, however you can explicitly
		define the functions as needed.
		
		Parameters:
			cls: Class type to register
			version (int): Version of class
			to (callable): Function for serializing class
			from_ (callable): Function for de-serializing class
			upgrade (callable): Function for upgrading class serialized data
		
		Returns:
			None
		'''
		
		if to is None:
			to = getattr(cls, "get_state_dict")
		if from_ is None:
			from_ = getattr(cls, "from_state_dict")
		if version is None:
			version = getattr(cls, "__schema_version__", 1)
		if upgrade is None:
			upgrade = getattr(cls, "upgrade", None)

		prev = SERIALIZABLE_CLASS_REGISTRY.get(cls.__name__)
		if prev and prev.cls is cls and prev.version == version and prev.to == to and prev.from_ == from_ and prev.upgrade == upgrade:
			return cls

		SERIALIZABLE_CLASS_REGISTRY[cls.__name__] = ClassRegistryInfo(
			cls=cls, to=to, from_=from_, version=version, upgrade=upgrade
		)
		return cls
	
	def __init_subclass__(cls, **kwargs):
		''' Automatically registers all subclasses in the class registry when the
		subclass is defined. 
		'''
		super().__init_subclass__(**kwargs)
		if cls is Serializable:
			return
		if not getattr(cls, "__auto_register__", True):
			return
		if not hasattr(cls, "__state_fields__"):
			raise AttributeError(f"{cls.__name__} must define __state_fields__")
		cls._register_json_class()
	
	@staticmethod
	def serialize(obj: Any) -> Any:
		
		def _encode_datetime(dt: datetime) -> dict:
			# Normalize to ISO 8601. Preserve tz if present; mark naivety explicitly.
			if dt.tzinfo is None:
				iso = dt.isoformat(timespec="microseconds")
				return {"__type__": "__datetime__", "data": iso, "naive": True}
			else:
				# Use RFC3339-style 'Z' when UTC to keep it compact.
				aware = dt.astimezone(timezone.utc)
				iso = aware.replace(tzinfo=timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
				return {"__type__": "__datetime__", "data": iso, "naive": False}

		def _encode_ndarray(arr) -> dict:
			# JSON-friendly: store shape/dtype/data (as nested lists).
			return {
				"__type__": "__ndarray__",
				"shape": list(arr.shape),
				"dtype": str(arr.dtype),
				"data": arr.tolist(),
			}
		
		def _is_numpy_scalar(o: Any) -> bool:
			return np is not None and isinstance(o, np.generic)
		
		def _is_numpy_array(o: Any) -> bool:
			return np is not None and isinstance(o, np.ndarray)
		
		# Registered custom classes
		info = SERIALIZABLE_CLASS_REGISTRY.get(type(obj).__name__)
		if info:
			payload = info.to(obj)
			return {
				"__type__": type(obj).__name__,
				"v": info.version,     # per-class schema version
				"data": Serializable.serialize(payload),
			}
		
		# datetime
		if isinstance(obj, datetime):
			return _encode_datetime(obj)
		
		# numpy
		if _is_numpy_scalar(obj):
			return obj.item()
		if _is_numpy_array(obj):
			return _encode_ndarray(obj)
		
		# Containers
		if isinstance(obj, dict):
			return {k: Serializable.serialize(v) for k, v in obj.items()}
		if isinstance(obj, (list, tuple)):
			return [Serializable.serialize(v) for v in obj]
		if isinstance(obj, set):
			return {"__type__": "__set__", "data": [Serializable.serialize(v) for v in obj]}
		
		# Primitives pass through
		return obj
	
	@staticmethod
	def deserialize(obj:Any) -> Any:
		
		def _decode_datetime(obj: dict) -> datetime:
			s = obj["data"]
			# Accept either ...Z or ...+HH:MM
			if s.endswith("Z"):
				s = s.replace("Z", "+00:00")
			dt = datetime.fromisoformat(s)
			# If it was marked naive originally, drop tzinfo
			if obj.get("naive"):
				return dt.replace(tzinfo=None)
			return dt

		def _decode_ndarray(obj: dict):
			if np is None:
				# Fall back to plain nested lists if numpy isn't available at load time
				return obj["data"]
			arr = np.array(obj["data"], dtype=obj["dtype"])
			return arr.reshape(tuple(obj["shape"]))
		
		# container-first walk
		if isinstance(obj, list):
			return [Serializable.deserialize(v) for v in obj]
		if isinstance(obj, dict):
			t = obj.get("__type__")
			if t == "__set__":
				return set(Serializable.deserialize(v) for v in obj["data"])
			if t == "__datetime__":
				return _decode_datetime(obj)
			if t == "__ndarray__":
				return _decode_ndarray(obj)

			# Registered classes
			if t and t in SERIALIZABLE_CLASS_REGISTRY:
				info = SERIALIZABLE_CLASS_REGISTRY[t]
				payload = Serializable.deserialize(obj["data"])  # decode inner payload first
				from_version = int(obj.get("v", 1))
				to_version = info.version
				if from_version != to_version:
					if info.upgrade:
						payload = info.upgrade(payload, from_version, to_version)
				return info.from_(payload)

			# Ordinary dict
			return {k: Serializable.deserialize(v) for k, v in obj.items()}

		# primitive
		return obj

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
			# print(f"{Fore.RED}Log object was not initialized in {type(self)} instance.!{Style.RESET_ALL}")
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