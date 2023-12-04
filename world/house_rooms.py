from dataclasses import dataclass

@dataclass
class living_room:
	name = "living_room"
	room_associations = {"dining_room": 0.8, "kitchen": 0.7, "utility_closet": 0.2, "bedroom": 0.6, "bathroom": 0.7}
	furniture_associations = {"couch": 0.9, "tv_set": 0.8, "bookshelf": 0.6, "coffee_table": 0.7, "chair": 0.7}
	object_associations = {"tv_remote": 0.8, "chess_board": 0.2, "magazine": 0.5, "laptop": 0.8, "painting": 0.7, "book": 0.8, "lamp": 0.9, "bicycle": 0.3}
	color = (21, 21, 203) # Red
	existence = 1.0
	duplicate = 0.0

@dataclass
class garage:
	room = "garage"
	room_associations = {"living_room": 0.1, "utility_closet": 0.8, "basement": 0.6}
	furniture_associations = {"fridge": 0.2, "shelf": 0.9}
	object_associations = {"wrench": 0.8, "drill": 0.8, "toolbox": 0.9, "milk": 0.2, "veggies": 0.1, "bicycle": 0.6, "trash_can": 0.2}
	color = (96, 96, 96) # Grey
	existence = 0.4
	duplicate = 0.0

@dataclass
class kitchen:
	room = "kitchen"
	room_associations = {"dining_room": 0.9, "living_room": 0.7, "pantry": 0.6, "bathroom": 0.7, "utility_closet": 0.1}
	furniture_associations = {"fridge": 1.0, "cupboard": 0.9, "counter_top": 0.9, "sink": 1.0, "dish_washer": 0.6, "drawers": 0.9, "liquor_cabinet": 0.3}
	object_associations = {"fork": 1.0, "plate": 1.0, "soap": 1.0, "spices": 1.0, "pot": 1.0, "cup": 1.0, "molasses": 0.3, "milk": 0.8, "veggies": 0.9, "vase": 0.4, "trash_can": 0.9}
	color = (0, 204, 102) # Green
	existence = 1.0
	duplicate = 0.0

@dataclass
class pantry:
	room = "pantry"
	room_associations = {"kitchen": 0.9}
	furniture_associations = {"shelf": 1.0}
	object_associations = {"rice": 0.6, "molasses": 0.4, "cereal": 0.8, "flour": 0.8, "spices": 0.9, "dog_food": 0.4}
	color = (255, 102, 178) # Lavendar
	existence = 0.9
	duplicate = 0.0

@dataclass
class bedroom:
	room = "bedroom"
	room_associations = {"dining_room": 0.7, "living_room": 0.7, "bathroom": 0.7}
	furniture_associations = {"bookshelf": 0.8, "bed": 1.0, "desk": 0.8, "nightstand": 0.9}
	object_associations = {"lamp": 1.0, "laptop": 0.7, "book": 0.8, "suit": 0.6, "chess_board": 0.4, "magazine": 0.3, "medicine": 0.5, "painting": 0.4, "trash_can": 0.2}
	color = (204, 102, 0) # Dark blue
	existence = 1.0
	duplicate = 0.8

@dataclass
class bathroom:
	room = "bathroom"
	room_associations = {"dining_room": 0.6, "kitchen": 0.7, "bedroom": 0.7, "living_room": 0.7, "basement": 0.2}
	furniture_associations = {"sink": 1.0, "cupboard": 0.9, "toilet": 1.0, "bathtub": 0.7, "shelf": 0.6}
	object_associations = {"toothbrush": 1.0, "toilet_paper": 1.0, "medicine": 0.8, "mouthwash": 0.8, "magazine": 0.6, "soap": 1.0, "shampoo": 0.9, "book": 0.1, "trash_can": 0.5}
	color = (254, 255, 243) # Eggshell
	existence = 1.0
	duplicate = 0.5

@dataclass
class basement:
	room = "basement"
	room_associations = {"bathroom": 0.4, "utility_closet": 0.6}
	furniture_associations = {"table": 0.9, "shelf": 0.6, "washing_machine": 0.6}
	object_associations = {"laundry_detergent": 0.9, "skis": 0.4, "vacuum_cleaner": 0.8, "bicycle": 0.4, "trash_can": 0.1}
	color = (105, 114, 60) # Turqoise grey
	existence = 0.3
	duplicate = 0.0

@dataclass
class utility_closet:
	room = "utility_closet"
	room_associations = {"basement": 0.5, "kitchen": 0.5, "living_room": 0.2}
	furniture_associations = {"shelf": 1.0}
	object_associations = {"mop": 0.9, "broom": 0.9, "dustpan": 0.9, "vacuum_cleaner": 0.5, "bleach": 0.6, "dog_food": 0.4, "toilet_paper": 0.3}
	color = (0, 51, 102) # Brown
	existence = 0.4
	duplicate = 0.1

@dataclass
class dining_room:
	room = "dining_room"
	room_associations = {"living_room": 0.9, "kitchen": 0.9, "bedroom":0.6, "bathroom": 0.3}
	furniture_associations = {"table": 1.0, "chair": 1.0, "liquor_cabinet": 0.5}
	object_associations = {"plate": 0.6, "cup": 0.7, "fork": 0.4, "vase": 0.7, "painting": 0.8, "wine": 0.8, "trash_can": 0.2}
	color = (255, 255, 0) # Cyan
	existence = 1.0
	duplicate = 0.0