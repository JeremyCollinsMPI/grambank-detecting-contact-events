from TreeFunctions import createTree, retain_only_nodes_that_are_in_list, findTips, find_glottocode, change_branch_length, findParent, findRoot,findChildren, findNodeNameWithoutStructure, findBranchLength
from PrepareWalsData import findStates
import pandas as pd
from Reconstruction import reconstructStatesForAllNodes
from LikelihoodFunction import calculateLikelihoodForAllNodes, findLikelihood, findTransitionProbability
from copy import deepcopy
import numpy as np
from pathlib import Path
import pickle

class Analysis:

    def __init__(self, load_from_file=False):
        self.rounding_to_nearest = 1
        self.load_from_file = load_from_file
        self.variables_to_store = ['trees', 'matrices', 'feature_states', 'feature_names']
    
    def run(self):
        if not self.load_from_file:
            self.prepare_without_loading_from_file()
        else: 
            self.prepare_with_loading_from_file()
    
    def prepare_without_loading_from_file(self):
        self.df = pd.read_csv('grambank-cldf/cldf/values.csv')
        self.make_trees()
        self.reconstruct_for_all_features()
        self.store_in_pickle_files()
    
    def prepare_with_loading_from_file(self):
        for variable_name in self.variables_to_store:
            with open('cache/' + variable_name + '.pkl', 'rb') as file:
                exec("self." + variable_name + " = pickle.load(file)")

    def make_trees(self):
        tree_strings = self.find_tree_strings()
        trees = [createTree(tree_string) for tree_string in tree_strings]
        grambank_languages = self.df['Language_ID'].unique()
        self.trees = [retain_only_nodes_that_are_in_list(tree, grambank_languages) for tree in trees]
        self.assign_node_heights_and_branch_lengths()

#         trees = json.load(open('reduced_trees.json', 'r'))
#         self.tree = [x for x in trees if  "'Bouyei [bouy1240][pcc]-l-':1" in x][0]
#         self.tree = retain_only_nodes_that_are_in_list(self.tree, grambank_languages)
        
    def find_tree_strings(self):
        return [
        '((jiam1236:1,(cunn1236:1,(bend1238:1,haaa1237:1,meif1236:1,qiii1237:1)hlai1239:1)nucl1241:1)hlai1238:1,((((emab1235:1,ennn1243:1,lang1316:1)nort2744:1,yero1238:1)buya1244:1,qabi1235:1)east2365:1,((baha1256:1,laha1250:1)paha1254:1,(((qauu1238:1,wanz1234:1)qaua1234:1,(((hong1242:1,aouu1238:1)aoua1234:1,redg1235:1)ahou1236:1,gela1264:1)nort3188:1,((duol1238:1,judu1234:1)whit1267:1,(sanc1245:1,wuch1237:1,zhen1240:1)gree1278:1)sout2749:1)gela1265:1,((lipu1242:1,lipu1241:1)lach1248:1,whit1266:1)lach1247:1)west2798:1)sout3143:1)kada1291:1,(((jizh1234:1,((chan1328:1,laoc1234:1,long1420:1,long1421:1,long1419:1,shis1234:1)qion1238:1,(huan1254:1,jial1234:1,linc1241:1,mani1307:1,meil1234:1,qiao1234:1,xiny1235:1)ling1262:1)ling1270:1)beic1239:1,(((minz1236:1,nong1247:1,(gian1240:1,khen1242:1,nung1284:1,nung1287:1,nung1285:1,nung1289:1,nung1286:1,nung1288:1,xuon1238:1)nung1283:1,yang1286:1)deba1238:1,(daiz1235:1,(((ahom1240:1,((taym1239:1)taid1247:1,thai1259:1)blac1269:1,(luuu1242:1,yong1277:1)luey1235:1,(taid1249:1,((nyaw1245:1,taid1252:1,taim1250:1)taid1248:1,taip1250:1)taim1254:1,tayt1241:1)redt1235:1,((aito1238:1,(assa1265:1,nort2738:1,sink1243:1)kham1290:1,kham1291:1,(phak1238:1,tail1248:1)phak1239:1,turu1249:1)assa1264:1,((koka1237:1,taim1251:1)shan1277:1,(deho1238:1,taip1249:1,yong1278:1)tain1252:1)nort2739:1,(khun1259:1,(band1336:1,nann1238:1,taiw1246:1)nort2740:1)sout2744:1,tail1247:1,(taih1246:1)unun9896:1)shan1276:1,taiy1242:1,(padi1241:1,taid1250:1,tait1248:1)whit1265:1)sout2743:1,(((laok1242:1,laok1241:1,luan1257:1,paks1238:1,sava1243:1,vien1239:1)laoo1244:1,((cent2011:1,nort2742:1,sout2745:1)nort2741:1,yoyy1238:1)sako1234:1,(takb1238:1,thai1260:1)sout2746:1,(khor1272:1)thai1261:1)laot1235:1,(phut1244:1,phua1239:1)siam1241:1)sput1235:1)sout3184:1,tays1238:1)sapa1255:1,(cent2012:1,east2364:1,nort2743:1,sout2747:1,tayb1242:1,tayt1242:1)tayy1238:1)wenm1239:1)cent2251:1,((((giay1234:1,hezh1245:1,qian1265:1,qian1260:1,qian1266:1)bouy1240:1,caol1238:1,cent2009:1,east2363:1,guib1245:1,guib1244:1,kuan1246:1,lian1252:1,liuj1238:1,liuq1235:1,qiub1238:1,tayk1238:1,tsun1242:1,(eeee1240:1)unun9894:1,youj1238:1)nort3189:1,(yong1276:1,yong1275:1)yong1274:1,zuoj1238:1)nort3180:1,(kham1289:1,naka1259:1)saek1240:1)nort3326:1)daic1237:1)daic1238:1,(((lian1256:1,(caom1238:1,nort2735:1)nort2734:1,naxi1247:1,sout2741:1)kami1255:1,mula1253:1)mula1252:1,((((boya1241:1,diee1238:1)aich1238:1,(chii1238:1,chin1463:1,hwaa1238:1,lyoo1238:1,makk1244:1)makc1235:1)maka1300:1,(chad1240:1,maon1241:1)maon1240:1,(anya1244:1,pand1262:1,sand1270:1)suii1243:1)maon1239:1,(hedo1238:1,hexi1238:1,huis1238:1)tenn1245:1)then1235:1)kams1241:1,((daga1277:1,biao1253:1,yong1283:1)biao1257:1,lakk1238:1)lakk1237:1)kamt1241:1)taik1256:1;',
        '((((kerd1240:1,keti1244:1,krau1243:1,kual1251:1,pula1260:1,uluc1240:1,ulut1240:1)jahh1242:1,(chew1245:1,(((ijoh1239:1,jaru1253:1,jehe1240:1,keda1249:1,kens1249:1,kens1250:1,kent1255:1,plus1242:1,ulus1242:1)kens1248:1,kint1239:1,(satu1239:1)tong1308:1)mani1291:1,(((bate1263:1,bate1266:1)bate1262:1,mint1239:1)bate1268:1,jede1239:1,(bate1267:1,nucl1294:1)jeha1242:1,minr1238:1)menr1235:1)mani1290:1)nort2682:1,((((lano1248:1,sabu1253:1)lano1247:1,semn1250:1)lano1246:1,(grik1245:1,kend1251:1,kene1242:1,lano1245:1,pokl1241:1,saka1282:1,sung1266:1,tanj1246:1,temb1269:1,uluk1255:1)temi1246:1)lano1244:1,(beta1249:1,bido1238:1,bill1238:1,came1250:1,jela1244:1,lipi1238:1,oran1256:1,pera1255:1,pera1254:1,sung1267:1,telo1238:1,uluk1256:1)sema1266:1)seno1278:1)cent1987:1,((beti1247:1,kual1250:1,mala1455:1,sela1257:1,sisi1247:1,ulul1240:1)besi1244:1,(sema1265:1,seme1247:1,temo1243:1)seme1246:1)sout2686:1)asli1243:1,(alak1253:1,(koll1253:1,traw1239:1)cuaa1241:1,((((alak1252:1,bahn1263:1,gola1254:1,jolo1240:1,kont1244:1,krem1240:1,tolo1253:1)bahn1262:1,mono1268:1)bona1256:1,((creq1239:1,nucl1300:1,raba1247:1)hree1244:1,(bahn1261:1,seda1260:1,west2400:1)reng1252:1,(cent1991:1,daks1239:1,grea1268:1,konh1238:1,kotu1238:1)seda1262:1)hres1237:1,todr1244:1)hres1236:1,katu1273:1,((hala1252:1,(jehb1240:1,jehm1239:1)jehh1245:1)jehh1246:1,kayo1245:1,(hala1253:1)unun9941:1)jehh1244:1,(kaco1239:1,roma1331:1)lama1291:1,taku1254:1)nort3150:1,(((chil1279:1,kalo1257:1,kodu1239:1,lacc1239:1,laya1252:1,nopp1240:1,pruu1239:1,rion1244:1,sopp1245:1,sree1244:1,tala1278:1,trin1270:1)koho1244:1,maaa1253:1)koho1243:1,((chal1265:1,chal1264:1,dorr1248:1,jroo1240:1,mroo1239:1,pran1244:1,tamu1245:1,vaji1239:1,voqt1239:1)chra1242:1,((chil1278:1,mnon1261:1,mnon1260:1,mnon1262:1)east2333:1,((biat1244:1,buda1251:1,buna1272:1,buru1295:1,dihb1240:1,preh1242:1)cent1992:1,krao1238:1,(buno1240:1,pran1245:1)sout2692:1)sout2691:1)mnon1259:1,(bude1236:1,(budi1247:1,bulo1243:1)bulo1242:1)stie1250:1)mnon1258:1)sout2690:1,tali1257:1,tamp1251:1,(lawi1235:1,((((hamo1235:1,kany1249:1,omba1245:1)nort3344:1,(jrii1235:1,kave1238:1)nort3343:1)nort3342:1,(krun1240:1,pala1337:1,lunb1239:1)sout3329:1)lave1249:1,(lave1248:1,souu1238:1)love1237:1,nyah1249:1,(innt1238:1,jeng1241:1,kran1247:1,riya1238:1,sokk1239:1,tama1324:1)oyyy1238:1,sapu1248:1)nucl1299:1,(trie1243:1)unun9940:1)west2399:1)bahn1264:1,(((east1236:1,west2398:1)nucl1297:1,phuo1238:1)katu1272:1,(pahi1245:1)paco1243:1,(ngeq1245:1,((hant1239:1,tong1309:1)lowe1395:1,ongg1239:1,(haaa1250:1,kamu1257:1,leem1238:1,pale1251:1,paso1238:1)uppe1406:1)ongt1234:1)taoi1247:1,((((brud1238:1,bruk1238:1,mang1379:1,trii1240:1)east2332:1,(nort3270:1,sout3251:1)kata1264:1)east2783:1,((soma1241:1,soph1238:1,sosl1238:1,sotr1238:1)sooo1254:1,west2397:1)west2870:1)brou1236:1,((chan1308:1,kuay1244:1,kuya1248:1,kuya1247:1,kuym1242:1,kuym1241:1,nheu1239:1)kuyy1240:1,nyeu1238:1)kuys1235:1)west1492:1)katu1271:1,(((((bhoi1239:1,cher1270:1,khyn1238:1,nucl1293:1)khas1269:1,((bara1355:1,bata1284:1,jowa1238:1,laka1251:1,mart1253:1,myns1238:1,nong1246:1,rali1240:1,rymb1238:1,shan1273:1,sume1240:1,sutn1238:1)jain1238:1,nong1245:1)pnar1238:1)khas1275:1,(mega1243:1,lyng1241:1)lyng1240:1)khas1274:1,(nucl1292:1,wark1246:1)warj1242:1)khas1268:1,(dana1252:1,((east2776:1,(huuu1240:1,manm1238:1,mokk1243:1,(doii1241:1,nucl1289:1)tail1246:1)sout3232:1,uuuu1243:1)angk1246:1,(bitt1240:1,(khan1275:1,khan1276:1)khan1274:1)khao1243:1,(lowe1393:1,uppe1404:1)lame1256:1,(((kemd1239:1,phan1252:1)blan1242:1,samt1238:1)bula1260:1,(((phal1255:1)east2330:1,(laoo1243:1)west2396:1)lawa1256:1,((awal1239:1,dama1281:1,masa1328:1)awac1238:1,para1301:1,xiyu1235:1,(ennn1242:1,kent1254:1,laaa1238:1,sonn1238:1,walo1238:1,wuuu1240:1)nucl1290:1)waaa1245:1)wala1271:1)waic1245:1)east2331:1,(((bule1241:1,raoj1238:1)ruch1235:1,shwe1236:1)pala1336:1,(rian1261:1,yinc1238:1)rian1260:1,ruma1248:1)west2791:1)pala1352:1)khas1273:1,((nucl1298:1,sout2688:1)cent1989:1,(buri1257:1,sisa1246:1,suri1265:1)nort2684:1,oldk1249:1)khme1253:1,(((hatt1237:1,khro1239:1,luan1256:1,lyyy1238:1,rokk1238:1,saya1243:1,uuuu1242:1,yuan1241:1)khmu1256:1,khue1238:1)khmu1255:1,mlab1235:1,(puoc1238:1,(oduu1239:1,phon1246:1,thee1239:1)pram1235:1,(mall1246:1,phai1238:1)tinn1239:1)phay1242:1)khmu1236:1,(mang1378:1,(boly1239:1,buga1247:1)boly1240:1)mang1377:1,((mata1282:1,pegu1239:1,yeee1239:1)monn1252:1,nyah1250:1,oldm1242:1)moni1258:1,((hill1251:1,plai1256:1)gata1239:1,((mund1322:1,mund1323:1)bodo1267:1,(lowe1394:1,uppe1405:1)bond1245:1)guto1244:1,juan1238:1,(dhel1238:1,dudh1239:1,mird1238:1)khar1287:1,(((((brij1240:1,manj1248:1)asur1254:1,bijo1238:1)asur1255:1,birh1242:1,((chai1235:1,loha1251:1)hooo1248:1,(bhum1234:1,hasa1250:1,kera1257:1,nagu1245:1,tamu1248:1)mund1320:1)homu1234:1,koda1236:1,(koda1254:1,(majh1254:1)korw1242:1)koda1256:1,majh1236:1,turi1246:1)mund1336:1,(kolb1241:1,maha1291:1,(kama1353:1,karm1240:1,loha1252:1,maha1292:1,manj1249:1,paha1253:1)sant1410:1)sant1457:1)kher1245:1,(bond1244:1,bour1248:1,mawa1263:1,ruma1249:1)kork1243:1)nort3151:1,(pare1266:1,(jura1242:1,sora1254:1)sora1256:1)sora1255:1)mund1335:1,(carn1240:1,(((camo1249:1,katc1248:1,nanc1247:1,trin1269:1)cent1990:1,(cond1241:1,grea1267:1,litt1242:1,milo1241:1,samb1303:1,tafw1239:1)sout2689:1)cent2305:1,(chau1258:1,(bomp1241:1)tere1272:1)chow1240:1)nucl1758:1)nico1262:1,(pear1247:1,(cent2314:1,chon1284:1,somr1240:1,(saoc1239:1,suoy1242:1)sout2684:1)west2394:1)pear1246:1,(((arem1240:1,(mayy1239:1,rucc1239:1,sach1240:1,sala1283:1)chut1247:1,(bola1249:1,khap1242:1,mali1278:1)male1282:1)chut1246:1,aheu1239:1)chut1252:1,((danl1238:1,lyha1238:1,phon1243:1,toum1239:1)hung1275:1,(cuoi1243:1,monn1251:1)thoo1240:1)cuoi1242:1,(((aota1239:1,boib1239:1,moii1253:1,moll1241:1,mual1240:1,than1254:1,wang1286:1)muon1246:1,nguo1239:1)muon1245:1,(cent1988:1,nort2683:1,sout2687:1)viet1252:1)viet1251:1)viet1250:1)aust1305:1;']

    def assign_node_heights_and_branch_lengths(self):
        import json
        self.node_heights_file = json.load(open('node_heights.json', 'r'))
        self.assign_node_heights()
        self.round_node_heights()
        self.assign_branch_lengths()
    
    def normalise_node_heights(self):
        self.find_largest_node_height()
        for key in self.node_heights_file.keys():
            self.node_heights_file[key] = self.node_heights_file[key] / self.largest_node_height
        
    def find_largest_node_height(self):
        if len(self.node_heights_file.values()) == 0:
            self.largest_node_height = None    
        self.largest_node_height = max(self.node_heights_file.values())
    
    def assign_node_heights(self):
        method = 'equidistant'
        for tree in self.trees:
            self.tree = tree
            for tip in findTips(tree):
                height = self.node_heights_file.get(findNodeNameWithoutStructure(tip))
                if height == None:
                    height = 0
                tree[tip]['height'] = height
            root = findRoot(tree)
            height = self.node_heights_file.get(findNodeNameWithoutStructure(root))
            if height == None:
                height = 1
            tree[root]['height'] = height
            for node in tree.keys():
                if not 'height' in tree[node].keys():
                    height = self.node_heights_file.get(findNodeNameWithoutStructure(node))
                    if not height == None:
                        tree[node]['height'] = height
            queue = []
            children = findChildren(root)
            for child in children:
                queue.append(child)
            done = False
            while not done:
                self.current_node = queue.pop()
                if not 'height' in tree[self.current_node].keys():
                    self.find_lower_bound_of_node_height()
                    self.find_upper_bound_of_node_height()
                    self.find_number_of_nodes_between_upper_and_lower_bounds_of_node_height()
                    amount_to_subtract = (self.upper_bound_of_node_height - self.lower_bound_of_node_height) / (self.number_of_nodes_between_upper_and_lower_bounds_of_node_height + 1)
                    amount_to_subtract = round(amount_to_subtract, 2)
                    tree[self.current_node]['height'] = self.upper_bound_of_node_height - amount_to_subtract
                children = findChildren(self.current_node)
                for child in children:
                    queue.append(child)                
                if len(queue) == 0:
                    done = True

    def find_lower_bound_of_node_height(self):
        queue = []
        heights = []
        numbers_of_nodes_between_upper_and_lower_bounds_of_node_height = []
        done = False
        current_node = self.current_node
        children = findChildren(current_node)
        for child in children:
            queue.append(child)
        while not done:
            current_node = queue.pop()
            if 'height' in self.tree[current_node].keys():
                heights.append(self.tree[current_node]['height'])
                number_of_nodes_away = 0
                number_of_nodes_away_process_finished = False
                p = current_node
                while not number_of_nodes_away_process_finished:
                    number_of_nodes_away += 1
                    p = findParent(self.tree, p)
                    if p == self.current_node:
                        number_of_nodes_away_process_finished = True
                numbers_of_nodes_between_upper_and_lower_bounds_of_node_height.append(number_of_nodes_away)
            else:
                children = findChildren(current_node)
                for child in children:
                    queue.append(child) 
            if len(queue) == 0:
                done = True
        self.lower_bound_of_node_height = max(heights)
        self.lower_bounds_of_node_height = heights
        self.numbers_of_nodes_between_upper_and_lower_bounds_of_node_height = numbers_of_nodes_between_upper_and_lower_bounds_of_node_height
            
    def find_upper_bound_of_node_height(self):
        parent = findParent(self.tree, self.current_node)
        self.upper_bound_of_node_height = self.tree[parent]['height']
    
    def find_number_of_nodes_between_upper_and_lower_bounds_of_node_height(self):
        self.number_of_nodes_between_upper_and_lower_bounds_of_node_height = self.numbers_of_nodes_between_upper_and_lower_bounds_of_node_height[self.lower_bounds_of_node_height.index(self.lower_bound_of_node_height)]
                
    def round_node_heights(self):
        for i in range(len(self.trees)):
            for node in self.trees[i].keys():
                self.trees[i][node]['height'] = round(self.trees[i][node]['height'] / self.rounding_to_nearest) * self.rounding_to_nearest

    def assign_branch_lengths(self):
        for i in range(len(self.trees)):
            tree = self.trees[i]
            new_tree = deepcopy(tree)
            root = findRoot(tree)
            queue = []
            done = False
            # Set branch length of root to zero
            new_tree = change_branch_length(new_tree, root, 0)
            children = findChildren(root)
            for child in children:
                queue.append([root, child])
            while not done:
                parent_and_child = queue.pop()
                parent = parent_and_child[0]
                child = parent_and_child[1]
                branch_length = tree[parent]['height'] - tree[child]['height']
                new_tree = change_branch_length(new_tree, child, branch_length)
                children = findChildren(child)
                current_child = child
                for child in children:
                    queue.append([current_child, child])
                if len(queue) == 0:
                    done = True
            self.trees[i] = deepcopy(new_tree)
                
    def reconstruct_for_all_features(self):
        self.matrices = {}
        self.feature_states = {}
        self.find_all_feature_names()
        for feature_name in self.feature_names[0:2]:
            self.feature_name = feature_name
            self.find_states()
            if '2' in self.states:
                continue
            for i in range(len(self.trees)):
                self.tree = self.trees[i]
                self.assign_feature_values_to_tips()
            self.find_most_likely_transition_probabilities()
            self.matrices[self.feature_name] = self.matrix
            self.feature_states[self.feature_name] = self.states
            for i in range(len(self.trees)):
                self.tree = self.trees[i]            
                self.reconstruct_values_given_matrix()
                self.trees[i] = deepcopy(self.tree)

    def find_all_feature_names(self):
        self.feature_names = self.df['Parameter_ID'].unique()

    def find_states(self):
        self.states = self.df['Value'][self.df['Parameter_ID'] == self.feature_name].unique().tolist()
        self.states = [x for x in self.states if not x == '?']

    def find_most_likely_transition_probabilities(self):
        current_matrix = None
        current_highest_likelihood = None
        rates_to_try = [0.9995, 0.9998, 0.9999, 0.99995, 0.99999]
        for rate in rates_to_try:
            matrix = [[rate, 1-rate], [1-rate, rate]]
            total_log_likelihood = 0
            for i in range(len(self.trees)):
                self.tree = self.trees[i]            
                likelihood = findLikelihood(self.tree, self.states, matrix, self.feature_name)
                total_log_likelihood = total_log_likelihood + np.log(likelihood)
            print(total_log_likelihood)
            if current_highest_likelihood == None:
                current_highest_likelihood = total_log_likelihood
                current_matrix = matrix
            elif total_log_likelihood > current_highest_likelihood:
                current_highest_likelihood = total_log_likelihood
                current_matrix = matrix
        self.matrix = current_matrix

    def reconstruct_values_given_matrix(self):
        self.tree = calculateLikelihoodForAllNodes(self.tree, self.states, self.matrix, self.feature_name)
        self.tree = reconstructStatesForAllNodes(self.tree, self.states, self.matrix, self.feature_name)
#         print(self.tree)
    
    def assign_feature_values_to_tips(self):
        states = self.states
        outputTree = self.tree
        tips = findTips(outputTree)
        for tip in tips:
            if outputTree[tip] == 'Unassigned':
                outputTree[tip] = {}
            glottocode = find_glottocode(tip)
            value_rows = self.df[(self.df['Language_ID'] == glottocode) & (self.df['Parameter_ID'] == self.feature_name)]
            if len(value_rows) == 0:
                value = '?'
            else:
                value = value_rows.iloc[0]['Value']
            outputTree[tip][self.feature_name] = {'states': {}}
            if value == '?':
                for state in states:
                    outputTree[tip][self.feature_name]['states'][state] = '?'
            else:
                for state in states:
                    if state == value:
                        outputTree[tip][self.feature_name]['states'][state] = 1
                    else:
                        outputTree[tip][self.feature_name]['states'][state] = 0
        self.tree = outputTree  

    def store_in_pickle_files(self):
        Path("cache").mkdir(parents=True, exist_ok=True)
        for variable_name in self.variables_to_store:
            with open('cache/' + variable_name + '.pkl', 'wb') as file:
                pickle.dump(eval("self." + variable_name), file)
    

if __name__ == "__main__":
    load_from_file = True
    instance = Analysis(load_from_file)
    instance.run()