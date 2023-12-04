from dataclasses import dataclass

@dataclass
class tv_remote:
    name = "tv_remote"
    furniture_associations = {"couch": 0.3, "tv_set": 0.9, "coffee_table": 0.5, "desk": 0.2}
    duplicate = 0.4

@dataclass
class chess_board:
    name = "chess_board"
    furniture_associations = {"coffee_table": 0.7, "bookshelf": 0.4, "nightstand": 0.2}
    duplicate = 0.1
    
@dataclass
class magazine:
    name = "magazine"
    furniture_associations = {"coffee_table": 0.8, "toilet": 0.5, "nightstand": 0.6, "bookshelf": 0.7}
    duplicate = 0.8
    
@dataclass
class wrench:
    name = "wrench"
    furniture_associations = {"shelf": 0.8}
    duplicate = 0.4
    
@dataclass
class drill:
    name = "drill"
    furniture_associations = {"shelf": 0.7}
    duplicate = 0.3

@dataclass
class fork:
    name = "fork"
    furniture_associations = {"drawers": 0.8, "counter_top": 0.8, "dish_washer": 0.5, "sink": 0.6, "table": 0.1}
    duplicate = 0.95

@dataclass
class plate:
    name = "plate"
    furniture_associations = {"cupboard": 0.9, "counter_top": 0.5, "dish_washer": 0.6, "sink": 0.7, "table": 0.4}
    duplicate = 0.95

@dataclass
class soap:
    name = "soap"
    furniture_associations = {"counter_top": 0.5, "sink": 0.9, "shelf": 0.3, "cupboard": 0.4}
    duplicate = 0.8

@dataclass
class pot:
    name = "pot"
    furniture_associations = {"cupboard": 0.8, "sink": 0.5, "counter_top": 0.1, "dish_washer": 0.2}
    duplicate = 0.9

@dataclass
class cup:
    name = "cup"
    furniture_associations = {"cupboard": 0.9, "counter_top": 0.7, "dish_washer": 0.7}
    duplicate = 0.95
    
@dataclass
class rice:
    name = "rice"
    furniture_associations = {"shelf": 1.0}
    duplicate = 0.2
    
@dataclass
class molasses:
    name = "molasses"
    furniture_associations = {"shelf": 0.8, "cupboard": 0.5}
    duplicate = 0.0
    
@dataclass
class cereal:
    name = "cereal"
    furniture_associations = {"cupboard": 0.5, "shelf": 0.8}
    duplicate = 0.3
    
@dataclass
class flour:
    name = "flour"
    furniture_associations = {"shelf": 0.9}
    duplicate = 0.1
    
@dataclass
class spices:
    name = "spices"
    furniture_associations = {"cupboard": 0.7, "counter_top": 0.6, "shelf": 0.8}
    duplicate = 0.2
    
@dataclass
class dog_food:
    name = "dog_food"
    furniture_associations = {"shelf": 0.9, "drawers": 0.4}
    duplicate = 0.1
    
@dataclass
class lamp:
    name = "lamp"
    furniture_associations = {"desk": 0.6, "bookshelf": 0.4, "nightstand": 0.7}
    duplicate = 0.4
    
@dataclass
class laptop:
    name = "laptop"
    furniture_associations = {"desk": 0.9, "nightstand": 0.6, "coffee_table": 0.3}
    duplicate = 0.1
    
@dataclass
class book:
    name = "book"
    furniture_associations = {"bookshelf": 1.0, "coffee_table": 0.6, "nightstand": 0.5, "bed": 0.3, "toilet": 0.1}
    duplicate = 0.9
    
@dataclass
class suit:
    name = "suit"
    furniture_associations = {"bed": 0.2}
    duplicate = 0.0
    
@dataclass
class toothbrush:
    name = "toothbrush"
    furniture_associations = {"sink": 0.9, "shelf": 0.6}
    duplicate = 0.6
    
@dataclass
class toilet_paper:
    name = "toilet_paper"
    furniture_associations = {"toilet": 0.95, "sink": 0.3, "shelf": 0.7}
    duplicate = 0.8
    
@dataclass
class medicine:
    name = "medicine"
    furniture_associations = {"sink": 0.7, "nightstand": 0.5}
    duplicate = 0.4
    
@dataclass
class mouthwash:
    name = "mouthwash"
    furniture_associations = {"sink": 0.9}
    duplicate = 0.2
    
@dataclass
class laundry_detergent:
    name = "laundry_detergent"
    furniture_associations = {"table": 0.2, "shelf": 0.6, "washing_machine": 0.8}
    duplicate = 0.3
    
@dataclass
class skis:
    name = "skis"
    furniture_associations = {"table": 0.4}
    duplicate = 0.1
    
@dataclass
class vacuum_cleaner:
    name = "vacuum_cleaner"
    furniture_associations = {}
    duplicate = 0.1
    
@dataclass
class mop:
    name = "mop"
    furniture_associations = {}
    duplicate = 0.0
    
@dataclass
class broom:
    name = "broom"
    furniture_associations = {}
    duplicate = 0.0
    
@dataclass
class dustpan:
    name = "dustpan"
    furniture_associations = {}
    duplicate = 0.0
    
@dataclass
class bleach:
    name = "bleach"
    furniture_associations = {"shelf": 0.7}
    duplicate = 0.2
    
@dataclass
class vase:
    name = "vase"
    furniture_associations = {"table": 0.5, "coffee_table": 0.3, "counter_top": 0.2}
    duplicate = 0.1

@dataclass
class shampoo:
    name = "shampoo"
    furniture_associations = {"bathtub": 0.9, "shelf": 0.3}
    duplicate = 0.6

@dataclass
class toolbox:
    name = "toolbox"
    furniture_associations = {"shelf": 0.8}
    duplicate = 0.2

@dataclass
class milk:
    name  = "milk"
    furniture_associations = {"fridge": 1.0}
    duplicate = 0.1

@dataclass
class veggies:
    name = "veggies"
    furniture_associations = {"fridge": 0.9, "counter_top": 0.4}
    duplicate = 0.3

@dataclass
class painting:
    name = "painting"
    furniture_associations = {}
    duplicate = 0.6

@dataclass
class wine:
    name = "wine"
    furniture_associations = {"liquor_cabinet": 1.0}
    duplicate = 0.7

@dataclass
class bicycle:
    name = "bicycle"
    furniture_associations = {}
    duplicate = 0.1

@dataclass
class trash_can:
    name = "trash_can"
    furniture_associations = {}
    duplicate = 0.7