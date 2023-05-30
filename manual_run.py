from star_analysis.service.alignment import AlignmentService
import os

basepath = os.path.join(os.path.dirname(__file__), 'data/images')
images = os.listdir(basepath)
images = [f"{basepath}/{i}" for i in images]
images = [i for i in images if os.path.isfile(i) and not i.endswith('jpg')]

optimal_allignement = AlignmentService().align_optimal(images)
optimal_allignement.shape
