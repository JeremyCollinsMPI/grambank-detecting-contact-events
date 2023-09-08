from TreeFunctions import createTree, retain_only_nodes_that_are_in_list, findTips, find_glottocode, change_branch_length, findParent, findRoot,findChildren, findNodeNameWithoutStructure, findBranchLength
from PrepareWalsData import findStates
import pandas as pd
from Reconstruction import reconstructStatesForAllNodes
from LikelihoodFunction import calculateLikelihoodForAllNodes, findLikelihood, findTransitionProbability
from copy import deepcopy
import numpy as np
from pathlib import Path
import pickle
import os

class Analysis:

    def __init__(self, load_from_file=False):
        self.rounding_to_nearest = 1
        self.load_from_file = load_from_file
        self.variables_to_store = ['trees', 'matrices', 'feature_states', 'features', 
            'trees_with_adjusted_branch_lengths', 'contact_events']
        self.df = pd.read_csv('grambank-cldf/cldf/values.csv')

    def run(self):
        if not self.load_from_file:
            self.prepare_without_loading_from_file()
        else: 
            self.prepare_with_loading_from_file()
        self.adjust_branch_lengths()
        self.infer_contact_events()
        self.analyse_contact_events()
    
    def prepare_without_loading_from_file(self):
        self.make_trees()
        self.reconstruct_for_all_features()
        self.store_in_pickle_files()
    
    def prepare_with_loading_from_file(self):
        for variable_name in self.variables_to_store:
            if variable_name + '.pkl' in os.listdir('cache'):
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
        '((((kerd1240:1,keti1244:1,krau1243:1,kual1251:1,pula1260:1,uluc1240:1,ulut1240:1)jahh1242:1,(chew1245:1,(((ijoh1239:1,jaru1253:1,jehe1240:1,keda1249:1,kens1249:1,kens1250:1,kent1255:1,plus1242:1,ulus1242:1)kens1248:1,kint1239:1,(satu1239:1)tong1308:1)mani1291:1,(((bate1263:1,bate1266:1)bate1262:1,mint1239:1)bate1268:1,jede1239:1,(bate1267:1,nucl1294:1)jeha1242:1,minr1238:1)menr1235:1)mani1290:1)nort2682:1,((((lano1248:1,sabu1253:1)lano1247:1,semn1250:1)lano1246:1,(grik1245:1,kend1251:1,kene1242:1,lano1245:1,pokl1241:1,saka1282:1,sung1266:1,tanj1246:1,temb1269:1,uluk1255:1)temi1246:1)lano1244:1,(beta1249:1,bido1238:1,bill1238:1,came1250:1,jela1244:1,lipi1238:1,oran1256:1,pera1255:1,pera1254:1,sung1267:1,telo1238:1,uluk1256:1)sema1266:1)seno1278:1)cent1987:1,((beti1247:1,kual1250:1,mala1455:1,sela1257:1,sisi1247:1,ulul1240:1)besi1244:1,(sema1265:1,seme1247:1,temo1243:1)seme1246:1)sout2686:1)asli1243:1,(alak1253:1,(koll1253:1,traw1239:1)cuaa1241:1,((((alak1252:1,bahn1263:1,gola1254:1,jolo1240:1,kont1244:1,krem1240:1,tolo1253:1)bahn1262:1,mono1268:1)bona1256:1,((creq1239:1,nucl1300:1,raba1247:1)hree1244:1,(bahn1261:1,seda1260:1,west2400:1)reng1252:1,(cent1991:1,daks1239:1,grea1268:1,konh1238:1,kotu1238:1)seda1262:1)hres1237:1,todr1244:1)hres1236:1,katu1273:1,((hala1252:1,(jehb1240:1,jehm1239:1)jehh1245:1)jehh1246:1,kayo1245:1,(hala1253:1)unun9941:1)jehh1244:1,(kaco1239:1,roma1331:1)lama1291:1,taku1254:1)nort3150:1,(((chil1279:1,kalo1257:1,kodu1239:1,lacc1239:1,laya1252:1,nopp1240:1,pruu1239:1,rion1244:1,sopp1245:1,sree1244:1,tala1278:1,trin1270:1)koho1244:1,maaa1253:1)koho1243:1,((chal1265:1,chal1264:1,dorr1248:1,jroo1240:1,mroo1239:1,pran1244:1,tamu1245:1,vaji1239:1,voqt1239:1)chra1242:1,((chil1278:1,mnon1261:1,mnon1260:1,mnon1262:1)east2333:1,((biat1244:1,buda1251:1,buna1272:1,buru1295:1,dihb1240:1,preh1242:1)cent1992:1,krao1238:1,(buno1240:1,pran1245:1)sout2692:1)sout2691:1)mnon1259:1,(bude1236:1,(budi1247:1,bulo1243:1)bulo1242:1)stie1250:1)mnon1258:1)sout2690:1,tali1257:1,tamp1251:1,(lawi1235:1,((((hamo1235:1,kany1249:1,omba1245:1)nort3344:1,(jrii1235:1,kave1238:1)nort3343:1)nort3342:1,(krun1240:1,pala1337:1,lunb1239:1)sout3329:1)lave1249:1,(lave1248:1,souu1238:1)love1237:1,nyah1249:1,(innt1238:1,jeng1241:1,kran1247:1,riya1238:1,sokk1239:1,tama1324:1)oyyy1238:1,sapu1248:1)nucl1299:1,(trie1243:1)unun9940:1)west2399:1)bahn1264:1,(((east1236:1,west2398:1)nucl1297:1,phuo1238:1)katu1272:1,(pahi1245:1)paco1243:1,(ngeq1245:1,((hant1239:1,tong1309:1)lowe1395:1,ongg1239:1,(haaa1250:1,kamu1257:1,leem1238:1,pale1251:1,paso1238:1)uppe1406:1)ongt1234:1)taoi1247:1,((((brud1238:1,bruk1238:1,mang1379:1,trii1240:1)east2332:1,(nort3270:1,sout3251:1)kata1264:1)east2783:1,((soma1241:1,soph1238:1,sosl1238:1,sotr1238:1)sooo1254:1,west2397:1)west2870:1)brou1236:1,((chan1308:1,kuay1244:1,kuya1248:1,kuya1247:1,kuym1242:1,kuym1241:1,nheu1239:1)kuyy1240:1,nyeu1238:1)kuys1235:1)west1492:1)katu1271:1,(((((bhoi1239:1,cher1270:1,khyn1238:1,nucl1293:1)khas1269:1,((bara1355:1,bata1284:1,jowa1238:1,laka1251:1,mart1253:1,myns1238:1,nong1246:1,rali1240:1,rymb1238:1,shan1273:1,sume1240:1,sutn1238:1)jain1238:1,nong1245:1)pnar1238:1)khas1275:1,(mega1243:1,lyng1241:1)lyng1240:1)khas1274:1,(nucl1292:1,wark1246:1)warj1242:1)khas1268:1,(dana1252:1,((east2776:1,(huuu1240:1,manm1238:1,mokk1243:1,(doii1241:1,nucl1289:1)tail1246:1)sout3232:1,uuuu1243:1)angk1246:1,(bitt1240:1,(khan1275:1,khan1276:1)khan1274:1)khao1243:1,(lowe1393:1,uppe1404:1)lame1256:1,(((kemd1239:1,phan1252:1)blan1242:1,samt1238:1)bula1260:1,(((phal1255:1)east2330:1,(laoo1243:1)west2396:1)lawa1256:1,((awal1239:1,dama1281:1,masa1328:1)awac1238:1,para1301:1,xiyu1235:1,(ennn1242:1,kent1254:1,laaa1238:1,sonn1238:1,walo1238:1,wuuu1240:1)nucl1290:1)waaa1245:1)wala1271:1)waic1245:1)east2331:1,(((bule1241:1,raoj1238:1)ruch1235:1,shwe1236:1)pala1336:1,(rian1261:1,yinc1238:1)rian1260:1,ruma1248:1)west2791:1)pala1352:1)khas1273:1,((nucl1298:1,sout2688:1)cent1989:1,(buri1257:1,sisa1246:1,suri1265:1)nort2684:1,oldk1249:1)khme1253:1,(((hatt1237:1,khro1239:1,luan1256:1,lyyy1238:1,rokk1238:1,saya1243:1,uuuu1242:1,yuan1241:1)khmu1256:1,khue1238:1)khmu1255:1,mlab1235:1,(puoc1238:1,(oduu1239:1,phon1246:1,thee1239:1)pram1235:1,(mall1246:1,phai1238:1)tinn1239:1)phay1242:1)khmu1236:1,(mang1378:1,(boly1239:1,buga1247:1)boly1240:1)mang1377:1,((mata1282:1,pegu1239:1,yeee1239:1)monn1252:1,nyah1250:1,oldm1242:1)moni1258:1,((hill1251:1,plai1256:1)gata1239:1,((mund1322:1,mund1323:1)bodo1267:1,(lowe1394:1,uppe1405:1)bond1245:1)guto1244:1,juan1238:1,(dhel1238:1,dudh1239:1,mird1238:1)khar1287:1,(((((brij1240:1,manj1248:1)asur1254:1,bijo1238:1)asur1255:1,birh1242:1,((chai1235:1,loha1251:1)hooo1248:1,(bhum1234:1,hasa1250:1,kera1257:1,nagu1245:1,tamu1248:1)mund1320:1)homu1234:1,koda1236:1,(koda1254:1,(majh1254:1)korw1242:1)koda1256:1,majh1236:1,turi1246:1)mund1336:1,(kolb1241:1,maha1291:1,(kama1353:1,karm1240:1,loha1252:1,maha1292:1,manj1249:1,paha1253:1)sant1410:1)sant1457:1)kher1245:1,(bond1244:1,bour1248:1,mawa1263:1,ruma1249:1)kork1243:1)nort3151:1,(pare1266:1,(jura1242:1,sora1254:1)sora1256:1)sora1255:1)mund1335:1,(carn1240:1,(((camo1249:1,katc1248:1,nanc1247:1,trin1269:1)cent1990:1,(cond1241:1,grea1267:1,litt1242:1,milo1241:1,samb1303:1,tafw1239:1)sout2689:1)cent2305:1,(chau1258:1,(bomp1241:1)tere1272:1)chow1240:1)nucl1758:1)nico1262:1,(pear1247:1,(cent2314:1,chon1284:1,somr1240:1,(saoc1239:1,suoy1242:1)sout2684:1)west2394:1)pear1246:1,(((arem1240:1,(mayy1239:1,rucc1239:1,sach1240:1,sala1283:1)chut1247:1,(bola1249:1,khap1242:1,mali1278:1)male1282:1)chut1246:1,aheu1239:1)chut1252:1,((danl1238:1,lyha1238:1,phon1243:1,toum1239:1)hung1275:1,(cuoi1243:1,monn1251:1)thoo1240:1)cuoi1242:1,(((aota1239:1,boib1239:1,moii1253:1,moll1241:1,mual1240:1,than1254:1,wang1286:1)muon1246:1,nguo1239:1)muon1245:1,(cent1988:1,nort2683:1,sout2687:1)viet1252:1)viet1251:1)viet1250:1)aust1305:1;',
        '((((clas1254:1,(((huan1253:1,padm1234:1,shan1295:1)rong1262:1,((chab1239:1,rebk1234:1,rtse1234:1)east2830:1,rong1261:1,(arik1266:1,mkha1234:1,rkan1234:1,them1242:1)nort3304:1,(rmag1234:1,abaa1238:1,sert1234:1)nort3305:1)hbro1238:1,(mdzo1234:1,rmas1234:1)inno1234:1,ndzo1234:1)amdo1237:1,((((dolp1239:1,tich1238:1)dolp1238:1,huml1238:1,(((gyal1236:1,(lhoo1238:1,namr1238:1,prok1238:1,sama1294:1)nubr1243:1)nubr1241:1,kyer1238:1)gyal1235:1,tsum1240:1,(kaga1252:1,(ilam1239:1,lamj1247:1,west2411:1,east2343:1)hela1238:1)yolm1234:1)kyir1235:1,lhom1239:1,(bara1356:1,uppe1408:1)lowa1242:1,(kara1471:1,muga1241:1)mugo1238:1,(jire1238:1,(naab1241:1,(khum1246:1,rame1238:1,solu1238:1)sher1255:1)sher1260:1)sher1254:1,walu1241:1)sout3216:1,((gart1234:1,gerg1239:1,pura1259:1,ruth1234:1,tsho1241:1)mnga1238:1,(ding1243:1,kyir1236:1,shig1238:1,west2905:1)gtsa1238:1,(drig1234:1,kong1283:1,utsa1239:1,lhok1239:1)dbus1238:1)tibe1272:1)cent2346:1,((nort2706:1,sout2713:1,west2412:1)baim1244:1,((koba1241:1,ther1234:1,dgon1234:1)hbru1241:1,palk1240:1,song1310:1,(byam1234:1,thew1237:1,thew1238:1,gser1235:1)thew1235:1,(nyin1251:1,thew1236:1)thew1234:1,(east2856:1,kluc1234:1,west2949:1)zhuo1234:1)chon1285:1,zhon1235:1)east2771:1,(((cham1333:1,dagy1234:1)cent2347:1,(amdo1238:1,bach1244:1,gert1234:1,horo1250:1,nakc1234:1)west2413:1,nort2707:1,(bath1249:1,dart1238:1,derg1234:1,kard1234:1,lith1253:1,mili1236:1,nyar1249:1)east2344:1,sout2714:1)kham1282:1,tsek1238:1)kham1299:1,((tukp1239:1,nako1240:1,nesa1234:1,pooh1234:1)bhot1235:1,((jadd1242:1,nila1245:1)jadd1243:1,spit1240:1)spit1239:1,(khok1238:1,maya1278:1,stod1242:1)stod1241:1)laha1255:1,(brok1248:1,brok1249:1,(choc1275:1,(lowe1398:1,uppe1409:1)grom1238:1,((haaa1251:1,nort2708:1,wang1287:1)dzon1239:1,laya1253:1,luna1243:1)nucl1307:1,sikk1242:1)dzon1238:1,lakh1240:1,(kham1283:1)uncl1513:1)sout3217:1)late1253:1)tibe1276:1,(((lalo1241:1,nyom1234:1,uppe1470:1)chan1309:1,lada1244:1,(cent2368:1,lowe1448:1,lung1254:1,uppe1471:1)zang1248:1)kenh1234:1,(balt1258:1,(puri1263:1,(nubr1242:1,sham1264:1)sham1283:1)puri1258:1)sham1282:1)lada1242:1)oldm1245:1,((((dakp1242:1,(khom1239:1)dzal1238:1)dakp1241:1,((((chog1238:1,chun1246:1,tang1332:1,uraa1244:1)bumt1240:1,(lowe1399:1,midd1326:1,uppe1410:1)khen1241:1,kurt1248:1)bumt1238:1,chal1267:1)chal1266:1,(chut1248:1,phob1239:1)nyen1254:1)phob1238:1)main1269:1,(olee1239:1,(nort3291:1,sout3272:1)east2810:1)olek1239:1)east1469:1,(kala1376:1,(bjok1234:1,dira1243:1,dung1256:1,mong1350:1,yabr1234:1)tsha1245:1)tsha1247:1)tsha1246:1,(basu1243:1)unun9962:1)bodi1257:1,((((bihi1238:1,chak1267:1,rana1247:1)kuta1241:1,(jaga1246:1,khor1243:1,nyak1258:1,phil1244:1,uiya1236:1)nort2709:1,(barp1238:1,kyau1238:1,lapr1238:1)sout2715:1)ghal1246:1,(((east2345:1,gork1242:1,lamj1244:1,tamu1246:1,(nort2710:1,sout2716:1)west2414:1)guru1261:1,((prak1243:1)mana1288:1,(narr1258:1,phuu1238:1)narp1239:1)mana1287:1,(chan1310:1,(marp1238:1,syan1241:1,tukc1238:1)thak1245:1)thak1244:1)guru1260:1,((kasi1252:1,kero1242:1)east2346:1,(cent2001:1,oute1247:1)east2347:1,(nort2711:1,rasu1238:1,sout2717:1,tris1240:1)west2415:1)nucl1729:1,(chuk1269:1,tang1333:1,teta1238:1)seke1240:1)tama1367:1)ghal1247:1,kaik1246:1)kaik1248:1,(((gahr1239:1,(marc1245:1,tolc1238:1)rong1264:1,suna1241:1,(zhan1239:1)unun9961:1)cent2311:1,((chau1259:1,((dhul1234:1,kuti1243:1,pang1283:1,yerj1238:1)byan1241:1,darm1243:1)darm1242:1)darm1241:1,rang1266:1)pith1234:1)east2777:1,((chit1279:1,kana1283:1,(kalp1234:1,nich1234:1,(razg1239:1,tukp1240:1)sang1346:1)kinn1249:1,(jang1254:1,shum1243:1)theb1237:1)kinn1250:1,((cent2002:1,cham1308:1,east2348:1)patt1248:1,tina1246:1)laha1249:1)west2868:1)tibe1275:1)bodi1256:1,(((((hari1249:1,nucl1318:1)dima1251:1,((debb1238:1,jama1260:1,noat1238:1,hala1254:1)kokb1239:1,rian1262:1,tipp1239:1,usui1238:1)tipp1238:1)dima1253:1,(((chot1240:1,mech1238:1)bodo1269:1,kach1279:1)bodo1280:1,tiwa1253:1)tiwa1256:1)boro1284:1,deor1238:1,(aben1248:1,achi1255:1,awee1241:1,aben1247:1,achi1247:1,chis1241:1,dacc1238:1,ganc1238:1,kamr1239:1,matc1241:1)garo1247:1,(aton1241:1,(bana1283:1,hari1248:1,satp1242:1,tint1241:1,wana1265:1)koch1250:1,(mait1251:1,rang1268:1)rabh1238:1,ruga1238:1)koch1249:1)bodo1279:1,((((ding1242:1,dulo1242:1)nort3289:1,(nump1234:1,turu1252:1)nort3288:1)sing1264:1,(enku1238:1,hkak1238:1,dzil1238:1,kaur1266:1,shid1239:1)kach1280:1)jing1260:1,(((bais1248:1,doch1234:1,naik1251:1)cakk1238:1,sakm1234:1)chak1270:1,(((andr1246:1,seng1275:1)andr1249:1,phay1241:1)andr1245:1,(gana1267:1,kadu1254:1)kado1242:1)chak1277:1,(burm1262:1,tama1328:1)uncl1508:1)sakk1239:1)jing1259:1,((chan1313:1,(khia1236:1,(lein1237:1,maky1235:1)lein1236:1)khia1235:1,(angp1238:1,chan1315:1,chen1256:1,chin1479:1,chin1480:1,choh1239:1,gele1238:1,hopa1238:1,jakp1238:1,kong1284:1,long1379:1,long1380:1,long1378:1,long1381:1,mohu1238:1,monn1253:1,mulu1242:1,ngan1293:1,sang1322:1,sham1276:1,shan1275:1,shen1248:1,shun1238:1,sima1257:1,sowa1243:1,tabl1242:1,tabu1242:1,tamk1238:1,tang1338:1,tobu1238:1,tola1246:1,toto1303:1)kony1248:1,(yong1271:1)phom1236:1,(borm1238:1,chan1314:1,horu1240:1,kulu1254:1)wanc1238:1)kony1247:1,(((hakh1237:1,hakh1236:1)kuwa1253:1,laju1238:1,(bote1240:1,hame1244:1,hasi1235:1,hath1235:1,khap1243:1,lama1296:1)nucl1786:1)noct1238:1,((have1240:1,mukl1235:1)mukl1234:1,((cham1335:1,lumn1234:1)cham1334:1,lang1339:1,nahe1239:1)nahe1238:1,((((hahc1235:1,ngem1252:1)hahc1234:1,(sang1326:1,yogl1238:1)jogl1234:1)jogl1235:1,(tong1312:1,long1382:1)long1418:1,(long1383:1,mosa1240:1)lung1253:1,mung1264:1,(kims1240:1,sank1250:1)shec1234:1)nort3332:1,(hkal1241:1,lang1315:1,miti1241:1,(higt1238:1,ronr1238:1)rera1241:1,rink1234:1)sout3320:1)tase1235:1,pont1258:1,(kato1246:1,keng1241:1,(lung1248:1,nokj1234:1,tikh1244:1,yong1272:1)nucl1787:1)tikh1243:1)tang1379:1,tuts1235:1)kony1249:1)kony1246:1,(chai1254:1)unun9960:1)brah1260:1,((((((lian1258:1,(long1373:1,main1270:1,xian1249:1)acha1249:1,luxi1238:1,ngoc1235:1)acha1252:1,(chas1234:1,lash1243:1)leqi1234:1,(lang1311:1,nucl1311:1,polo1241:1)zaiw1241:1)high1273:1,(nort2721:1,sout2727:1)hpon1238:1,((dago1244:1,gawa1248:1,hlol1236:1,laki1245:1,lawn1238:1,wakh1246:1,zaga1238:1)maru1249:1,pela1242:1)midn1240:1)nort2720:1,((danu1251:1,inth1239:1)inth1238:1,((marm1234:1,rakh1245:1)arak1255:1,((boma1245:1,mand1476:1,yaww1238:1)nucl1310:1,oldb1235:1)oldm1246:1,(dawe1238:1,merg1238:1,pala1339:1)tavo1242:1)nucl1730:1,taun1248:1)sout3159:1)burm1266:1,(((((bisu1244:1,laom1237:1,pyen1239:1)bisu1246:1,(coon1239:1,(blac1268:1,hwet1239:1,khas1272:1,mung1263:1,whit1264:1)phun1245:1)phun1244:1,sang1320:1,(bant1298:1,cauh1234:1)uncl1502:1)biso1241:1,((biyo1243:1,enuu1235:1,kadu1253:1,mpii1239:1)bika1252:1,((akeu1235:1,(akoo1247:1,ason1245:1)akha1245:1,chep1243:1,muda1235:1)akha1246:1,(angl1266:1,dazh1234:1,gehu1234:1,guoh1234:1,guoz1234:1,khab1234:1,lami1239:1,luob1237:1,luom1234:1)hani1248:1)haya1251:1,(baih1238:1)honi1244:1)hani1250:1,phan1254:1,(cosa1234:1,sila1247:1)sila1251:1)biso1244:1,(buyu1238:1,youl1235:1)jino1236:1)hani1249:1,(kuco1235:1,(lahu1254:1,naaa1244:1,nyii1239:1,sheh1238:1)lahu1253:1,lahu1252:1)laho1234:1,((kats1235:1,sadu1234:1,sama1295:1)kazh1234:1,(sout2719:1,lawu1238:1)lawo1234:1,(((lipo1242:1,miqi1235:1)lipo1244:1,(nanh1240:1,shua1253:1,yaoa1238:1)lolo1259:1,(hler1235:1,limi1243:1,mili1235:1)unun9959:1)lipo1243:1,(((((kuan1249:1,kuam1234:1)kuan1252:1,sona1244:1)kuan1251:1,(ekaa1234:1,(((west1506:1,xish1235:1)cent2297:1,dong1286:1,(east2696:1)uncl1503:1)core1258:1,xuzh1234:1)grea1292:1,sout3210:1,yang1304:1)lalo1240:1,talu1238:1)lalu1234:1,(hual1240:1,(dech1234:1,ning1281:1)blac1267:1,nort3298:1,lush1248:1)lisu1250:1)lisu1252:1,tang1372:1)nucl1734:1,(lamu1257:1,nalu1239:1,sama1296:1)unun9958:1)liso1234:1,((azha1235:1,((nort2714:1,sout2721:1)sani1269:1,(axiy1235:1,azhe1235:1)sani1275:1)sani1267:1)sani1266:1,(((lagh1245:1,((((bokh1237:1,phum1235:1)bokh1236:1,muzi1235:1,(nort2716:1,sout2722:1)nort2715:1)nucl1309:1,qila1235:1)core1246:1,thop1236:1)thop1235:1)lagh1244:1,moji1238:1)muji1235:1,((anip1235:1,labo1243:1)anil1235:1,(hlep1236:1,(khlu1236:1,zokh1238:1)khlu1235:1,phuk1235:1)hlep1235:1)phow1235:1)high1272:1,(((farn1234:1,(east2349:1,(esha1238:1,xinp1238:1)nort2718:1)nort2717:1,sout2723:1)nisu1238:1,nyis1235:1)nisu1237:1,((((gepo1234:1,same1240:1,sani1265:1,(ayiz1244:1,ches1238:1)unun9955:1,(luqu1238:1,tazh1234:1,wudi1239:1)wudi1238:1)nasu1237:1,((henk1238:1,hezh1242:1,wein1239:1)wume1235:1,(biji1244:1,dafa1241:1,qian1262:1)wusa1235:1)nesu1235:1)nesu1234:1,(butu1242:1,huil1243:1,nort2713:1,yinu1238:1,yish1238:1)sich1238:1)nasu1236:1,awuu1235:1,(aluo1235:1)unun9956:1)nucl1739:1,(ache1244:1)uncl1517:1)niso1234:1,(((phup1239:1,(daba1263:1,suob1234:1)phuz1235:1)phup1238:1,(alug1235:1,phup1237:1)phup1236:1)down1239:1,(phal1256:1,(alop1235:1,phol1237:1)phol1236:1)upri1239:1)rive1256:1,(nisi1238:1)unun9957:1)sout3212:1)nili1235:1,((cent2003:1,nort2719:1,sout2724:1)nusu1239:1,(guol1234:1,wupi1238:1)zauz1238:1)nuso1234:1,phol1235:1)lolo1267:1,((maan1239:1,mang1429:1,maza1306:1,muan1234:1,(mant1265:1,mond1267:1)munj1248:1)mond1268:1,kath1251:1)mond1269:1,(kokc1239:1,suph1238:1)ugon1239:1,(pail1244:1)unun9953:1)lolo1265:1,(((duox1238:1,lisu1245:1,nucl1312:1)ersu1241:1,lizu1234:1,tosu1234:1)ersu1242:1,guiq1238:1,((laze1238:1,(lata1234:1,yong1288:1)yong1270:1,(lapa1246:1,lich1241:1,luti1241:1)naxi1245:1)nais1236:1,(east2350:1,west2416:1)namu1246:1,shix1238:1)naic1235:1,(((((dats1234:1,gdon1234:1)japh1234:1,tsho1240:1,(sida1238:1)zbua1234:1)jiar1240:1,(jinc1238:1,lixi1238:1,maer1238:1,xiao1244:1)situ1238:1)core1262:1,(((dgeb1234:1,(east2851:1,west2941:1)gesh1238:1,phos1234:1,daof1238:1)horp1239:1,(dayi1243:1,puxi1242:1,zong1241:1)shan1274:1)horp1240:1,(eree1240:1,erga1238:1,guan1252:1,muer1238:1,siya1242:1,taiy1241:1,xiao1245:1,yelo1242:1)guan1266:1)horp1241:1)rgya1241:1,(east2351:1,(nort3389:1,sout3361:1)west2417:1)muya1239:1,((jisu1234:1,labo1245:1,mudi1234:1,sany1234:1,taob1238:1,tuoq1234:1,taob1239:1,zuos1234:1)nort2723:1,((renh1234:1,xich1234:1,xiny1234:1)cent2357:1,(daya1246:1,qing1238:1)west2926:1)sout2729:1)pumi1242:1,(((daji1238:1,(taop1238:1,tong1330:1)east2793:1)down1240:1,(long1374:1,mian1253:1)inwa1234:1)sout2728:1,((cimu1238:1,heih1238:1,jiao1241:1,luhu1242:1,sout3258:1,(sanl1247:1,weig1238:1,yadu1238:1)west2876:1)nort2722:1,(gouk1234:1,yong1287:1)sout3257:1)upst1234:1)qian1264:1,quey1238:1,tang1334:1,((drag1234:1,(lame1270:1,lamo1246:1,gser1234:1)lamo1245:1,zlar1234:1)cham1336:1)uncl1511:1,zhab1238:1)qian1263:1)naqi1236:1,((baoj1238:1,long1377:1)nort2732:1,sout2739:1)tuji1244:1)burm1265:1,((east2362:1,west2036:1)dhim1246:1,toto1302:1)dhim1245:1,(idum1241:1,diga1241:1)mish1241:1,gong1251:1,((ilam1238:1,reng1254:1,tams1238:1)lepc1244:1,lhok1238:1,(((bujh1238:1,(east2353:1,west2419:1)chep1245:1)chep1244:1,dura1244:1,((ghus1238:1,tama1326:1)gama1251:1,((bhuj1238:1)east2354:1,(luku1238:1,taka1261:1,thab1238:1,wale1244:1)west2420:1)parb1234:1,(jang1256:1,tapn1238:1)shes1236:1)kham1286:1,(east2352:1,west2418:1)maga1261:1)kham1285:1,((((chuk1270:1,(chha1251:1,kham1288:1,maha1294:1,naml1238:1,pelm1238:1,pidi1240:1,sota1241:1,tama1327:1)kulu1253:1,(bang1337:1,dima1250:1,heda1238:1,khar1289:1,para1303:1,rakh1246:1)nach1240:1)kulu1252:1,(bung1263:1,samb1304:1)saam1282:1,(bhal1246:1,halu1238:1,khar1290:1,khot1254:1,phal1257:1,sama1297:1,tana1278:1,tong1311:1)samp1249:1)kham1300:1,(((east2359:1,(nort2729:1,sout2737:1)inte1256:1,west2423:1)bant1281:1,wali1261:1)bant1280:1,caml1239:1,(khes1241:1)dung1252:1,puma1239:1)sout2736:1)cent2250:1,(((athp1241:1,belh1239:1)athp1240:1,chhu1238:1,chhi1245:1,(east2358:1,nort2728:1,sout2735:1)yakk1236:1)grea1285:1,(((biks1238:1)nort2727:1,((gess1238:1,paoo1246:1,siba1240:1)sout2734:1,yamp1242:1)yamp1244:1)loho1238:1,((dibu1238:1,mulg1243:1,suns1242:1)east2357:1,(bala1299:1,bumd1238:1)west2422:1)mewa1252:1)uppe1412:1)east2719:1,(chat1267:1,pant1253:1,phed1238:1,tapl1238:1)limb1266:1,(((balk1255:1,madh1243:1,ratn1236:1)jeru1240:1,(bonu1238:1,ubuu1238:1)wamb1257:1)chau1260:1,((bane1244:1,dobo1243:1,namb1290:1,proc1238:1,rokh1238:1)bahi1252:1,(sure1238:1)sunw1242:1,wayu1241:1)nort2730:1,(cent2007:1,east2360:1,lann1238:1,nort2731:1,sout2738:1)thul1246:1,(chos1238:1,doru1238:1)tilu1238:1,((bras1238:1,khar1291:1,lamd1238:1,makp1238:1)dumi1241:1,khal1275:1,(behe1240:1,sung1268:1)koii1238:1)uppe1413:1)west2424:1)kira1253:1,(((badi1252:1,(dolk1238:1,doti1235:1,jeth1235:1,sind1275:1,tota1238:1)dola1240:1)east2773:1,(bakt1238:1,kath1252:1,(bagl1238:1,citl1238:1)west2965:1)newa1246:1)newa1247:1,(bara1357:1,(east2355:1,sind1274:1,west2421:1)than1259:1)than1258:1)newa1245:1)maha1306:1)hima1249:1,(((bwek1238:1,geba1237:1)geba1236:1,((east2342:1,west2409:1)kaya1317:1,yint1235:1)kaya1337:1,(brek1238:1,manu1255:1)kaya1316:1,mobw1234:1)cent1999:1,(geko1235:1,kaya1315:1,zaye1235:1,yinb1236:1)nort2703:1,((nort2705:1,sout2712:1)paok1235:1,(((kanc1245:1,kawk1239:1,paan1239:1,ratc1238:1,tavo1241:1)pwoe1235:1,(maub1238:1,tuan1238:1)pwow1235:1)east2341:1,(phra1235:1,(maep1238:1,maes1238:1,omko1238:1)pwon1235:1)nort2704:1)pwoo1239:1)peri1254:1,((bili1249:1,derm1238:1)paku1238:1,((pala1338:1,pana1290:1)sgaw1245:1,wewa1238:1)sgaw1244:1)sout1554:1)kare1337:1,((bich1234:1,dikh1234:1,kasp1234:1,namp1239:1,sing1271:1,wang1301:1)bugu1246:1,(((bugu1245:1,lasu1234:1)chay1250:1,kuru1311:1,sari1249:1)sulu1241:1,(bulu1255:1,kojo1234:1,rawa1269:1)west2873:1)puro1234:1,((chug1252:1,lish1235:1)chug1251:1,((jeri1243:1,khoi1253:1,khoi1252:1,rahu1234:1)sart1249:1,(rupa1234:1,sher1261:1)sher1257:1)sher1256:1)meyi1234:1)khob1235:1,(miju1243:1,zakh1243:1)gema1234:1,((((((chak1268:1,dzun1241:1,kehe1240:1,khon1248:1,kohi1249:1,mima1238:1,mozo1238:1,nali1243:1,teng1262:1,teny1242:1)anga1288:1,chok1243:1)anga1287:1,khez1235:1,((paom1238:1)maon1238:1,poum1235:1)naga1397:1)anga1244:1,(nort2725:1,poch1243:1)poch1242:1,((azon1238:1,kete1251:1)sout2732:1,(daya1244:1,laze1239:1,zhim1238:1,zumo1240:1)sumi1235:1)reng1253:1)anga1286:1,((chan1311:1,chon1286:1,dord1238:1,long1376:1,mong1332:1,(teng1273:1,yach1235:1)yach1234:1)aona1235:1,(kyoo1238:1,kyon1245:1,kyon1246:1,kyou1238:1,live1239:1,ndre1238:1,tson1248:1)loth1237:1,(kiza1238:1,phel1241:1,phot1238:1,pirr1238:1,purr1238:1,thuk1239:1)sang1321:1,((long1375:1,maku1273:1)maku1272:1,para1302:1,(chir1282:1,(mini1250:1,pher1238:1,yimc1241:1)sout2731:1,tikh1241:1,waii1241:1)yimc1240:1)yimc1239:1)aoic1235:1)anga1312:1,(amri1238:1,(chin1477:1,mirl1238:1,rong1265:1)karb1241:1)karb1240:1,(((darl1242:1,tawr1235:1,((bawm1236:1,bual1235:1,fala1243:1)fala1242:1,(klan1242:1,shon1249:1,zokh1239:1)haka1240:1)laic1236:1,((biet1238:1,hmar1241:1,(hade1238:1)hran1239:1,(khel1238:1,sake1246:1,than1257:1)saka1283:1)hmar1240:1,(fann1238:1,leee1238:1,mizo1245:1,ngen1250:1,ralt1243:1,tlau1238:1)lush1249:1)mizo1244:1,pank1249:1)cent2005:1,((laut1236:1,(hlaw1238:1,tlon1239:1)mara1382:1,shen1247:1,(lowe1400:1,uppe1411:1)zyph1238:1)nucl1757:1,sent1260:1,zotu1235:1)mara1381:1)cent2330:1,(((lang1312:1)aimo1244:1,syri1242:1)aimo1247:1,(laiz1241:1,muls1238:1)anal1239:1,chir1283:1,chot1239:1,(bong1301:1,bong1300:1,chor1276:1,dabb1234:1,hran1240:1,kaip1242:1,kalo1267:1,lank1243:1,mols1234:1,mors1234:1,rang1271:1,rupi1234:1)rang1267:1,koir1240:1,(kolh1240:1)komi1270:1,lamk1238:1,(mons1234:1,moyo1238:1)moyo1240:1,sorb1250:1,tara1313:1)oldk1252:1,((east2779:1,(kami1254:1,kham1284:1,khim1238:1,khum1249:1,khun1258:1,khwe1238:1,kumi1246:1,ngal1290:1,yind1246:1)khum1248:1,mroc1235:1,reng1255:1)khom1240:1,(((gang1266:1,vaip1239:1)gang1271:1,ngaw1239:1,simt1238:1,siyi1240:1,zouu1235:1)siza1239:1,(khar1288:1,(bukp1238:1,dapz1238:1,dimm1239:1,dimp1238:1,lamz1238:1,lous1238:1,telz1238:1,tuic1245:1)pait1244:1,puru1266:1,ralt1242:1,(kamh1239:1,sokt1238:1)tedi1235:1,(bait1247:1,chan1312:1,hawk1238:1,jang1255:1,kaok1238:1,khon1237:1,kipg1238:1,lang1313:1,sair1238:1,shit1241:1,sing1263:1,than1256:1)thad1238:1)thad1239:1)nort3179:1,(((chit1280:1,khya1239:1,lemy1239:1,minb1243:1,sain1244:1,sand1269:1,thay1247:1)asho1236:1,sumt1234:1)asho1237:1,(chin1478:1,((kanp1238:1,matu1256:1,pale1252:1)daai1236:1,(nitu1238:1)munc1235:1)daai1235:1,(halt1238:1,thui1238:1,tlam1238:1,vala1243:1)ngal1291:1)choi1241:1)sout3160:1)peri1260:1)kuki1246:1,(loii1241:1,meit1246:1,pang1284:1)mani1292:1,((khoi1251:1,mari1416:1)mari1415:1,((kupo1240:1,phad1238:1,ukhr1238:1)tang1336:1,nort3287:1,nort3286:1,khan1277:1,(suan1234:1)uncl1516:1)sino1246:1)tang1335:1,((mara1380:1,ngat1245:1,tkhu1238:1,will1242:1)mara1379:1,(lian1251:1,mzie1235:1,(njau1238:1,pare1267:1)zeme1240:1)nucl1313:1,(song1297:1)rong1266:1,than1255:1)zeme1241:1)kuki1245:1,((((biji1251:1,egab1234:1,enqi1234:1,nuji1238:1,luob1236:1,tuol1234:1)nort2724:1,((eryu1239:1,heqi1238:1,jian1239:1,lanp1241:1,yunl1238:1)cent2004:1,((qili1234:1,zhou1234:1)dali1242:1,xian1250:1)sout2730:1)sout3254:1)baic1239:1,caij1234:1)caij1235:1,long1417:1)macr1275:1,((koro1316:1,mila1245:1)koro1317:1,(((kark1255:1,komk1238:1,pasi1253:1,shim1250:1)bori1243:1,(miny1239:1,miri1271:1,miny1240:1,pada1257:1)misi1242:1)east2361:1,(((pang1285:1)damu1236:1,tang1377:1)damu1237:1,(apat1240:1,(ashi1243:1,pail1243:1,ramo1243:1)boka1249:1,(((bang1338:1)naaa1245:1,tagi1241:1)bang1372:1,galo1242:1,(akal1238:1,hill1258:1,nish1241:1)nyis1236:1)suba1255:1)west2797:1)prew1234:1)tani1259:1)macr1268:1,(bang1369:1,(east2847:1,west2937:1)saja1240:1)miji1239:1,((anuu1243:1,hkon1236:1)anuu1241:1,mruu1242:1)mrui1235:1,(((dulo1243:1,nuri1244:1)drun1238:1,(byab1238:1,chol1277:1,gwaz1238:1,kizo1238:1,miko1239:1,nora1239:1)nung1282:1)guno1235:1,(aguu1241:1,hpun1238:1,htis1238:1,kunl1238:1,long1385:1,matw1238:1,mutw1238:1,nucl1319:1,serh1238:1,serw1238:1,tang1339:1,taro1261:1,wada1259:1,wahk1238:1,zith1239:1)rawa1265:1)nung1293:1,(raji1240:1,(raut1239:1,rawa1264:1)raut1240:1)raji1239:1,((lite1248:1,(((anyi1247:1,nanc1254:1,ping1246:1)chan1317:1,fugu1238:1,jich1238:1,yili1247:1,ying1246:1)ganc1239:1,(hail1247:1,huiz1243:1,yuet1238:1,ning1269:1,raop1234:1,(liud1234:1,taoy1234:1)sanh1239:1,(chan1325:1,lian1259:1,ming1255:1,ning1278:1,qing1239:1,shan1292:1,wupi1239:1,yong1284:1)ting1250:1,tong1314:1,yueb1238:1,yuez1238:1,yugu1248:1)hakk1236:1,(comm1247:1,zhon1237:1)midd1344:1,((huhe1234:1,ping1247:1,taiy1243:1)jiny1235:1,(daoh1239:1,hezh1244:1,((beij1234:1,taib1240:1)beij1235:1,(xian1253:1,xini1234:1,zhen1241:1)huab1238:1,(hefe1234:1,nanj1234:1)jing1262:1,(jina1245:1,tian1238:1)jilu1239:1,(lanz1234:1,wulu1243:1,yinc1239:1)xibe1241:1,ning1279:1,(haer1234:1,shen1252:1)nort3283:1,qing1240:1,rong1271:1,(chen1267:1,guiy1236:1,kunm1234:1,wuha1234:1)xina1239:1,yang1306:1,ying1249:1,yuci1234:1)mand1415:1,tang1373:1,wutu1241:1,((ganz1245:1,shaa1242:1,yage1238:1)dung1253:1,gang1272:1)zhon1236:1)mand1471:1)nort3155:1,quji1234:1,((jing1261:1,(hong1243:1,hong1244:1,shex1234:1,tang1380:1)jixi1238:1,qide1238:1,(huan1251:1,sout3255:1,tunx1234:1,wuyu1234:1,xiun1234:1,yixi1234:1)xiuy1238:1,yanz1241:1)huiz1242:1,((long1386:1,qing1241:1,chuz1238:1)chuq1241:1,jinh1238:1,((lins1238:1,(ning1280:1,zhou1235:1)yong1273:1)nort3273:1,(hang1257:1,pili1238:1)nort3274:1,(jiax1234:1,shan1293:1,suzh1234:1,wuxi1234:1)suhu1238:1,(huzh1239:1,sout3256:1)tiao1238:1)taih1244:1,taiz1238:1,ouji1238:1,wuzh1238:1,(shil1262:1,taig1246:1,tong1313:1)xuan1238:1)wuch1236:1)wuhu1234:1,((chan1326:1,yiyu1234:1,yuey1234:1)chan1316:1,(baoj1239:1,chen1269:1,guzh1234:1,huay1242:1,jish1242:1,luxi1239:1,xupu1234:1,yuan1243:1)chen1268:1,heng1239:1,(loud1234:1,shua1258:1,xian1254:1)luos1238:1,(dong1298:1,jian1243:1,quan1243:1)yong1285:1)xian1251:1,(((dong1299:1,guil1241:1)nort3268:1,(nann1239:1,nann1240:1,upri1240:1)sout3250:1)ping1245:1,((gaoz1234:1,yang1307:1)gaol1235:1,guin1237:1,qinl1235:1,(danc1234:1,jian1244:1,tois1237:1,vanc1237:1)siyi1236:1,wuhu1235:1,yong1286:1,((guan1279:1,xian1255:1,tank1239:1,wuzh1239:1,xigu1234:1)cant1236:1,guan1280:1,samy1234:1,zhon1239:1)yueh1236:1)yuec1235:1)yuep1234:1)midd1354:1)clas1255:1,(((funi1234:1,fuzh1239:1)mind1253:1,((chao1238:1,shan1244:1)chao1239:1,dati1239:1,((chan1329:1,hain1237:1,wann1243:1,wenc1234:1,yaxi1234:1)hain1238:1,leiz1236:1)qion1239:1,(chae1235:1,taib1242:1,xiam1236:1,zhan1240:1,fuji1236:1)hokk1242:1,zhen1239:1,(long1252:1,nanl1234:1,sanx1234:1)zhon1238:1)minn1241:1,(puti1238:1,xian1252:1)puxi1243:1)coas1318:1,(((chon1289:1,jian1240:1,jian1241:1,song1309:1,zhen1242:1)minb1236:1,(jian1242:1,shao1235:1,shun1239:1)shao1234:1)minb1244:1,minz1235:1)inla1267:1)minn1248:1,(late1251:1,shan1294:1)oldc1244:1,waxi1236:1)sini1245:1,(lure1234:1,namm1235:1)unun9951:1)sino1245:1;'
        ]

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
        self.find_all_features()
        for feature in self.features:
            self.feature = feature
            self.find_states()
            if '2' in self.states:
                index = np.where(self.features == feature)[0][0] 
                self.features = np.delete(self.features, index)
                print('deleted ' + self.feature)
                continue
            for i in range(len(self.trees)):
                self.tree = self.trees[i]
                self.assign_feature_values_to_tips()
            self.find_most_likely_transition_probabilities()
            self.matrices[self.feature] = self.matrix
            self.feature_states[self.feature] = self.states
            print(self.feature)
            for i in range(len(self.trees)):
                self.tree = self.trees[i]            
                self.reconstruct_values_given_matrix()
                self.trees[i] = deepcopy(self.tree)

    def find_all_features(self):
        self.features = self.df['Parameter_ID'].unique()

    def find_states(self):
        if self.feature in self.feature_states.keys():
            self.states = self.feature_states[self.feature]
            return
        self.states = self.df['Value'][self.df['Parameter_ID'] == self.feature].unique().tolist()
        self.states = [x for x in self.states if not x == '?']

    def find_most_likely_transition_probabilities(self):
        current_matrix = None
        current_highest_likelihood = None
        rates_to_try = [0.9995, 0.9998, 0.9999, 0.99995, 0.99999]
        for rate_1 in rates_to_try:
            for rate_2 in rates_to_try:
                matrix = [[rate_1, 1-rate_1], [1-rate_2, rate_2]]
                total_log_likelihood = 0
                for i in range(len(self.trees)):
                    self.tree = self.trees[i]            
                    likelihood = findLikelihood(self.tree, self.states, matrix, self.feature)
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
        self.tree = calculateLikelihoodForAllNodes(self.tree, self.states, self.matrix, self.feature)
        self.tree = reconstructStatesForAllNodes(self.tree, self.states, self.matrix, self.feature)
    
    def assign_feature_values_to_tips(self):
        states = self.states
        outputTree = self.tree
        tips = findTips(outputTree)
        for tip in tips:
            if outputTree[tip] == 'Unassigned':
                outputTree[tip] = {}
            glottocode = find_glottocode(tip)
            value_rows = self.df[(self.df['Language_ID'] == glottocode) & (self.df['Parameter_ID'] == self.feature)]
            if len(value_rows) == 0:
                value = '?'
            else:
                value = value_rows.iloc[0]['Value']
            outputTree[tip][self.feature] = {'states': {}}
            if value == '?':
                for state in states:
                    outputTree[tip][self.feature]['states'][state] = '?'
            else:
                for state in states:
                    if state == value:
                        outputTree[tip][self.feature]['states'][state] = 1
                    else:
                        outputTree[tip][self.feature]['states'][state] = 0
        self.tree = outputTree  

    def store_in_pickle_files(self):
        Path("cache").mkdir(parents=True, exist_ok=True)
        for variable_name in self.variables_to_store:
            with open('cache/' + variable_name + '.pkl', 'wb') as file:
                try:
                    pickle.dump(eval("self." + variable_name), file)
                except:
                    pass

    def adjust_branch_lengths(self):
        if 'trees_with_adjusted_branch_lengths.pkl' in os.listdir('cache'):
            self.trees = self.trees_with_adjusted_branch_lengths
        else:
            self.make_trees_with_adjusted_branch_lengths()
            self.reconstruct_features_again()
            self.trees_with_adjusted_branch_lengths = self.trees
            self.store_in_pickle_files()
    
    def make_trees_with_adjusted_branch_lengths(self):
        for i in range(len(self.trees)):
            tree = self.trees[i]
            self.tree = tree
            self.new_tree = deepcopy(self.tree)
            for node in tree.keys():
                self.node = node
                self.parent = findParent(tree, node)
                if self.parent == None:
                    continue
                self.adjust_branch_length_for_node() 
            self.trees[i] = deepcopy(self.new_tree)
        # not adjusting node heights for now        
        self.need_to_adjust_node_heights_after_adjusting_branch_lengths = False
        if self.need_to_adjust_node_heights_after_adjusting_branch_lengths:
            self.adjust_tree_heights()
  
    def adjust_branch_length_for_node(self):
        self.branch_length = float(findBranchLength(self.node))
        self.original_branch_length = self.branch_length
        self.calculate_likelihood_of_node_given_parent_for_adjusting_branch_length()
        self.current_highest_likelihood = self.likelihood_of_node_given_parent_for_adjusting_branch_length
        self.try_adjusting_branch_length_downwards()
        self.try_adjusting_branch_length_upwards()
        print('New branch length: ', self.branch_length)
        node_name_without_structure = findNodeNameWithoutStructure(self.node)
        for node in self.new_tree.keys():
            if findNodeNameWithoutStructure(node) == node_name_without_structure:
                node_name = node
                break
        self.new_tree = change_branch_length(self.new_tree, node_name, self.branch_length)

    def calculate_likelihood_of_node_given_parent_for_adjusting_branch_length(self):
        likelihood = 0
        for feature in self.features:
            probability = 0
            self.feature = feature
            self.find_states()
            self.matrix = self.matrices[self.feature]
            for state2 in self.states:
                if self.tree[self.node][self.feature]['reconstructedStates'][state2] == '?':
                    continue
                for state1 in self.states:
                    transition_probability = findTransitionProbability(
                        state1, state2, self.states, self.matrix, self.branch_length)
                    probability += self.tree[self.parent][self.feature]['reconstructedStates'][state1] \
                        * transition_probability * self.tree[self.node][self.feature]['reconstructedStates'][state2]
            likelihood = likelihood + np.log(probability)
            self.likelihood_of_node_given_parent_for_adjusting_branch_length = likelihood

    def try_adjusting_branch_length_downwards(self):
        likelihood_has_gone_down = False
        while not likelihood_has_gone_down:
            self.branch_length = self.branch_length - 100
            if self.branch_length <= 0:
                self.branch_length = self.branch_length + 100
                return
            self.calculate_likelihood_of_node_given_parent_for_adjusting_branch_length()
            if self.likelihood_of_node_given_parent_for_adjusting_branch_length > self.current_highest_likelihood:
                self.current_highest_likelihood = self.likelihood_of_node_given_parent_for_adjusting_branch_length
            else:
                likelihood_has_gone_down = True
                self.branch_length = self.branch_length + 100
                return

    def try_adjusting_branch_length_upwards(self):
        maximum_allowed_branch_length = 5000
        likelihood_has_gone_down = False
        while not likelihood_has_gone_down:
            if self.branch_length >= maximum_allowed_branch_length:
                self.branch_length = self.original_branch_length
                return
            self.branch_length = self.branch_length + 100
            self.calculate_likelihood_of_node_given_parent_for_adjusting_branch_length()
            if self.likelihood_of_node_given_parent_for_adjusting_branch_length > self.current_highest_likelihood:
                self.current_highest_likelihood = self.likelihood_of_node_given_parent_for_adjusting_branch_length
            else:
                likelihood_has_gone_down = True
                self.branch_length = self.branch_length - 100
                return
    
    def reconstruct_features_again(self):
        for feature in self.features:
            self.feature = feature
            self.find_states()
            for i in range(len(self.trees)):
                self.tree = self.trees[i]
                self.assign_feature_values_to_tips()
            self.matrix = self.matrices[self.feature]
            self.states = self.feature_states[self.feature] 
            print(self.feature)
            for i in range(len(self.trees)):
                self.tree = self.trees[i]            
                self.reconstruct_values_given_matrix()
                self.trees[i] = deepcopy(self.tree)

    def infer_contact_events(self):
        self.initialise_borrowing_probabilities()
        if not 'contact_events.pkl' in os.listdir('cache'):
            self.contact_events = []
            self.different_family_contact_events = []
            for tree in self.trees:
                self.tree = tree
                for node in tree.keys():
                    print('-----')
                    print('Node 1: ' + node)
                    self.node = node
                    self.parent = findParent(tree, node)
                    if self.parent == None:
                        print('root, excluding for now')
                        continue
                    self.find_contact_events_for_node()
                    self.select_one_contact_event_for_node()
            self.sort_contact_events_by_likelihood_difference()
            self.store_in_pickle_files()
        # self.adjust_borrowing_probabilities()
    
    def initialise_borrowing_probabilities(self):
        initial = 0.1
        self.borrowing_probabilities = {}
        for feature in self.features:
            states = self.feature_states[feature]
            self.borrowing_probabilities[feature] = {}
            for state in states:
                self.borrowing_probabilities[feature][state] = initial
                
    def find_contact_events_for_node(self):
        self.contact_events_for_node = []
        for tree in self.trees:
            self.compared_tree = tree
            self.find_contemporary_lineage_nodes()
            for node2 in self.contemporary_lineage_nodes:
                self.feature_evidences = {}
                self.node2 = node2
                likelihoods = []
                for contact_intensity in range(0, 3):
                    self.contact_intensity = contact_intensity
                    self.find_likelihood_of_transition_from_parent_to_current_node()
                    likelihoods.append(self.likelihood)
                most_likely_contact_intensity = likelihoods.index(max(likelihoods))
                likelihood_difference = max(likelihoods) - likelihoods[0]
                if most_likely_contact_intensity > 0:
                    self.sort_features_explained_by_contact_by_evidence()
                    if self.node_1_and_node_2_are_from_different_families():
                        self.find_ancestral_probs()
                        self.contact_events_for_node.append(
                            {
                                'node_1': self.node,
                                'node_2': self.node2,
                                'contact_intensity': most_likely_contact_intensity,
                                'likelihood': max(likelihoods),
                                'features_better_explained_by_contact': deepcopy(self.features_better_explained_by_contact),
                                'tree_1_index': self.trees.index(self.tree),
                                'tree_2_index': self.trees.index(self.compared_tree), 
                                'evidence_for_contact': likelihood_difference
                            })
   
    def select_one_contact_event_for_node(self):
        if len(self.contact_events_for_node) > 0:
            maximum_likelihood = max([x['likelihood'] for x in self.contact_events_for_node])
            to_append = [x for x in self.contact_events_for_node if x['likelihood'] == maximum_likelihood][0]
            self.contact_events.append(to_append)
                                                        
    def find_contemporary_lineage_nodes(self):
        self.contemporary_lineage_nodes = []
        for node in self.compared_tree.keys():
            self.node2 = node
            if self.node2_is_contemporary_to_node():
                self.contemporary_lineage_nodes.append(self.node2)
    
    def node2_is_contemporary_to_node(self):
        if self.node2 == self.node:
            return False
        if self.compared_tree[self.node2]['height'] <= self.tree[self.node]['height']:
            return False
        children = findChildren(self.node2)
        if self.node in children:
            return False
        for child in children:
            if self.compared_tree[child]['height'] > self.tree[self.node]['height']:
                return False
        return True
    
    def find_likelihood_of_transition_from_parent_to_current_node(self):
        total_log_likelihood = 0
        self.features_better_explained_by_contact = []
        for feature in self.features:
            self.feature = feature
            self.find_states()
            self.matrix = self.matrices[self.feature]
            self.find_likelihood_of_transition_from_parent_to_current_node_for_feature()
            total_log_likelihood += np.log(self.likelihood_of_transition_from_parent_to_current_node_for_feature)
        self.likelihood = total_log_likelihood
        
    def find_likelihood_of_transition_from_parent_to_current_node_for_feature(self):
        branch_length = float(findBranchLength(self.node))
        likelihood = 0
        likelihood_if_there_was_no_borrowing = 0
        for state2 in self.states:
            if self.tree[self.node][self.feature]['reconstructedStates'][state2] == '?':
                continue
            probability_that_contact_node_had_state_2 = self.compared_tree[self.node2][self.feature]['reconstructedStates'][state2]
            borrowing_probability = self.borrowing_probabilities[self.feature][state2]
            probability_that_it_is_not_borrowed = (1 - borrowing_probability) ** self.contact_intensity
            probability_that_it_is_borrowed = 1 - probability_that_it_is_not_borrowed
            probability_under_borrowing = probability_that_contact_node_had_state_2 \
                * probability_that_it_is_borrowed * self.tree[self.node][self.feature]['reconstructedStates'][state2]
            probability_under_no_borrowing = 0
            for state1 in self.states:
                transition_probability = findTransitionProbability(
                    state1, state2, self.states, self.matrix, branch_length)
                probability_under_no_borrowing += self.tree[self.parent][self.feature]['reconstructedStates'][state1] \
                    * probability_that_it_is_not_borrowed * transition_probability * self.tree[self.node][self.feature]['reconstructedStates'][state2]
                likelihood_if_there_was_no_borrowing += self.tree[self.parent][self.feature]['reconstructedStates'][state1] \
                    * transition_probability * self.tree[self.node][self.feature]['reconstructedStates'][state2]
            probability = probability_under_borrowing + probability_under_no_borrowing
            likelihood = likelihood + probability
        self.likelihood_of_transition_from_parent_to_current_node_for_feature = likelihood
        self.likelihood_if_there_was_no_borrowing = likelihood_if_there_was_no_borrowing
        self.update_feature_evidences()
        self.update_features_better_explained_by_contact()

    def update_feature_evidences(self):
        difference_in_probability_assuming_borrowing = np.log(self.likelihood_of_transition_from_parent_to_current_node_for_feature / self.likelihood_if_there_was_no_borrowing)
        if not self.feature in self.feature_evidences:
            self.feature_evidences[self.feature] = []
        self.feature_evidences[self.feature].append(difference_in_probability_assuming_borrowing)

    def update_features_better_explained_by_contact(self):
        values_of_node_1 = self.tree[self.node][self.feature]['reconstructedStates']
        value_of_node_1 = [z[0] for z in list(values_of_node_1.items()) if z[1] == max(list(values_of_node_1.values()))][0]
        values_of_node_2 = self.compared_tree[self.node2][self.feature]['reconstructedStates']
        value_of_node_2 = [z[0] for z in list(values_of_node_2.items()) if z[1] == max(list(values_of_node_2.values()))][0]
        values_of_parent = self.tree[self.parent][self.feature]['reconstructedStates']
        value_of_parent = [z[0] for z in list(values_of_parent.items()) if z[1] == max(list(values_of_parent.values()))][0]
        prob_for_node_1 = values_of_node_1[value_of_node_1]
        prob_for_node_2 = values_of_node_2[value_of_node_1]
        prob_for_parent = values_of_parent[value_of_node_1]
        if prob_for_node_1 > prob_for_parent and prob_for_node_2 > prob_for_parent:
            self.features_better_explained_by_contact.append({'name': self.feature, 
                'value_of_node_1': value_of_node_1,
                'prob_for_node_1': prob_for_node_1, 'prob_for_node_2': prob_for_node_2,
                'prob_for_parent': prob_for_parent,
                'evidence_for_borrowing': self.feature_evidences[self.feature]})
        elif prob_for_node_1 < prob_for_parent and prob_for_node_2 < prob_for_parent:
            self.features_better_explained_by_contact.append({'name': self.feature, 
                'value_of_node_1': value_of_node_1,
                'prob_for_node_1': prob_for_node_1, 'prob_for_node_2': prob_for_node_2,
                'prob_for_parent': prob_for_parent,
                'evidence_for_borrowing': self.feature_evidences[self.feature]})
     
    def node_1_and_node_2_are_siblings(self):
        children = findChildren(self.parent)
        if self.node2 in children:
            return True
        else:
            return False
    
    def node_1_and_node_2_are_from_different_families(self):
        if self.tree == self.compared_tree:
            return False
        else:
            return True
    
    def sort_features_explained_by_contact_by_evidence(self):
        self.features_better_explained_by_contact = sorted(self.features_better_explained_by_contact, 
            key=lambda x: max(x['evidence_for_borrowing']), reverse=True)
    
    def find_ancestral_probs(self):
        for i in range(len(self.features_better_explained_by_contact)):
            self.feature = self.features_better_explained_by_contact[i]['name']
            values_of_node_1 = self.tree[self.node][self.feature]['reconstructedStates']
            self.value_of_node_1 = [z[0] for z in list(values_of_node_1.items()) if z[1] == max(list(values_of_node_1.values()))][0]
            self.features_better_explained_by_contact[i]['node_1_ancestral_probs'] = self.find_node_1_ancestral_probs()
            self.features_better_explained_by_contact[i]['node_2_ancestral_probs'] = self.find_node_2_ancestral_probs()

    def find_node_1_ancestral_probs(self):
        ancestral_probs = []
        x = self.parent
        while not x == None:
            ancestral_probs.append(self.tree[x][self.feature]['reconstructedStates'][self.value_of_node_1])
            x = findParent(self.tree, x)
        return ancestral_probs

    def find_node_2_ancestral_probs(self):
        ancestral_probs = []
        x = findParent(self.compared_tree, self.node2)
        while not x == None:
            ancestral_probs.append(self.compared_tree[x][self.feature]['reconstructedStates'][self.value_of_node_1])
            x = findParent(self.compared_tree, x)
        return ancestral_probs

    def sort_contact_events_by_likelihood_difference(self):
        self.contact_events = sorted(self.contact_events, key=lambda x: x['evidence_for_contact'], reverse=True)

    def adjust_borrowing_probabilities(self):
        numbers_to_try = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for feature in self.features:
            self.feature = feature
            self.find_states()
            for state in self.states:
                total_likelihoods = []
                for number_to_try in numbers_to_try:
                    total_likelihood = 0
                    for contact_event in self.contact_events:
                        self.tree = self.trees[contact_event['tree_1_index']]
                        self.compared_tree = self.trees[contact_event['tree_2_index']]
                        self.matrix = self.matrices[self.feature]
                        self.borrowing_probabilities[self.feature][state] = number_to_try
                        self.node = contact_event['node_1']
                        self.parent = findParent(self.tree, self.node)
                        self.node2 = contact_event['node_2']
                        self.contact_intensity = contact_event['contact_intensity']
                        likelihood = self.calculate_likelihood_for_finding_borrowing_probability()
                    total_likelihood = total_likelihood + np.log(likelihood)
                    total_likelihoods.append(total_likelihood)
                new_borrowing_probability = numbers_to_try[total_likelihoods.index(max(total_likelihoods))]
                self.borrowing_probabilities[self.feature][state] = new_borrowing_probability
        
    def calculate_likelihood_for_finding_borrowing_probability(self):
        branch_length = float(findBranchLength(self.node))
        likelihood = 0
        for state2 in self.states:
            if self.tree[self.node][self.feature]['reconstructedStates'][state2] == '?':
                continue
            probability_that_contact_node_had_state_2 = self.compared_tree[self.node2][self.feature]['reconstructedStates'][state2]
            borrowing_probability = self.borrowing_probabilities[self.feature][state2]
            probability_that_it_is_not_borrowed = (1 - borrowing_probability) ** self.contact_intensity
            probability_that_it_is_borrowed = 1 - probability_that_it_is_not_borrowed
            probability_under_borrowing = probability_that_contact_node_had_state_2 \
                * probability_that_it_is_borrowed * self.tree[self.node][self.feature]['reconstructedStates'][state2]
            probability_under_no_borrowing = 0
            for state1 in self.states:
                transition_probability = findTransitionProbability(
                    state1, state2, self.states, self.matrix, branch_length)
                probability_under_no_borrowing += self.tree[self.parent][self.feature]['reconstructedStates'][state1] \
                    * probability_that_it_is_not_borrowed * transition_probability * self.tree[self.node][self.feature]['reconstructedStates'][state2]
            probability = probability_under_borrowing + probability_under_no_borrowing
            likelihood = likelihood + probability
        return likelihood

    def analyse_contact_events(self):
        for contact_event in self.contact_events:
            print(contact_event['node_1'])
            print(contact_event['node_2'])
            print(np.exp(contact_event['evidence_for_contact']))
            print(contact_event['features_better_explained_by_contact'][0:3])
            print('---------------')

if __name__ == "__main__":
    load_from_file = True
    instance = Analysis(load_from_file)
    instance.run()