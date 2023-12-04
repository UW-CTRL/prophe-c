from dataclasses import dataclass

# COLORS ARE BGR

@dataclass
class couch:
	name = "couch"
	object_associations = {}
	duplicate = 0.3
	color = (165, 100, 65)

@dataclass
class tv_set:
	name = "tv_set"
	object_associations = {}
	duplicate = 0.0
	color = (97, 95, 94)

@dataclass
class bookshelf:
	name = "bookshelf"
	object_associations = {}
	duplicate = 0.1
	color = (44, 44, 100)

@dataclass
class coffee_table:
	name = "coffee_table"
	object_associations = {}
	duplicate = 0.1
	color = (190, 189, 174)
				
@dataclass
class fridge:
	name = "fridge"
	object_associations = {}
	duplicate = 0.2
	color = (228, 228, 228)
				
@dataclass
class chair:
	name = "chair"
	object_associations = {}
	duplicate = 0.8
	color = (4, 70, 113)

@dataclass
class cupboard:
	name = "cupboard"
	object_associations = {}
	duplicate = 0.9
	color = (96, 121, 137)

@dataclass
class counter_top:
	name = "counter_top"
	object_associations = {}
	duplicate = 0.5
	color = (245, 245, 245)

@dataclass
class dish_washer:
	name = "dish_washer"
	object_associations = {}
	duplicate = 0.0
	color = (182, 182, 182)

@dataclass
class bed:
	name = "bed"
	object_associations = {}
	duplicate = 0.2
	color = (67, 126, 15)
	
@dataclass
class desk:
	name = "desk"
	object_associations = {}
	duplicate = 0.1
	color = (66, 94, 133)
	
@dataclass
class nightstand:
	name = "nightstand"
	object_associations = {}
	duplicate = 0.2
	color = (49, 69, 111)
	
@dataclass
class toilet:
	name = "toilet"
	object_associations = {}
	duplicate = 0.0
	color = (250, 250, 250)
	
@dataclass
class bathtub:
	name = "bathtub"
	object_associations = {}
	duplicate = 0.0
	color = (250, 250, 250)
	
@dataclass
class shelf:
	name = "shelf"
	object_associations = {}
	duplicate = 0.7
	color = (126, 139, 165)
	
@dataclass
class washing_machine:
	name = "washing_machine"
	object_associations = {}
	duplicate = 0.0
	color = (250, 250, 250)
	
@dataclass
class table:
	name = "table"
	object_associations = {}
	duplicate = 0.0
	color = (10, 22, 117)

@dataclass
class sink:
	color = (250, 250, 250)

@dataclass
class drawers:
	color = (90, 48, 66)

@dataclass
class liquor_cabinet:
	color = (19, 69, 139)