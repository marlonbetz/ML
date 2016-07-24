import numpy as np

SimilarityTestData = [
                        #check for voiceless plosive > nasal via unvoiced plosive 
                        {"positive":"p n".split(),"negative":"t".split(),"true":"m".split()},
                        {"positive":"t m".split(),"negative":"p".split(),"true":"n".split()},
                        {"positive":"k n".split(),"negative":"t".split(),"true":"N".split()},
                        #check for voiced plosive > nasal via voiced plosive 
                        {"positive":"b n".split(),"negative":"d".split(),"true":"m".split()},
                        {"positive":"d m".split(),"negative":"b".split(),"true":"n".split()},
                        {"positive":"g n".split(),"negative":"d".split(),"true":"N".split()},
                        
                        #check for voiceless plosive > nasal via voiced plosive 
                        {"positive":"p n".split(),"negative":"d".split(),"true":"m".split()},
                        {"positive":"t m".split(),"negative":"b".split(),"true":"n".split()},
                        {"positive":"k n".split(),"negative":"d".split(),"true":"N".split()},
                        #check for voiced plosive > nasal via unvoiced plosive 
                        {"positive":"b n".split(),"negative":"t".split(),"true":"m".split()},
                        {"positive":"d m".split(),"negative":"p".split(),"true":"n".split()},
                        {"positive":"g n".split(),"negative":"t".split(),"true":"N".split()},
                        
                        #check unvoiced plosive > voiced plosive via voiced plosive
                        {"positive":"p d".split(),"negative":"t".split(),"true":"b".split()},
                        {"positive":"t b".split(),"negative":"p".split(),"true":"d".split()},
                        {"positive":"k d".split(),"negative":"t".split(),"true":"g".split()},
                        
                        #check voiced plosive > unvoiced plosive via unvoiced plosive                    
                        {"positive":"b t".split(),"negative":"d".split(),"true":"p".split()},
                        {"positive":"d p".split(),"negative":"b".split(),"true":"t".split()},
                        {"positive":"g t".split(),"negative":"d".split(),"true":"k".split()},
                        
                        #check unvoiced pulmonic plosive  > unvoiced aspirated plosive via unvoiced aspirated plosive                    
                        {"positive":"p th~".split(),"negative":"t".split(),"true":"ph~".split()},
                        {"positive":"t ph~".split(),"negative":"p".split(),"true":"th~".split()},
                        {"positive":"k th~".split(),"negative":"t".split(),"true":"kh~".split()},
                        
                        #check voiced pulmonic plosive  > voiced aspirated plosive via voiced aspirated plosive                    
                        {"positive":"b dh~".split(),"negative":"d".split(),"true":"bh~".split()},
                        {"positive":"d bh~".split(),"negative":"b".split(),"true":"dh~".split()},
                        {"positive":"g dh~".split(),"negative":"d".split(),"true":"gh~".split()},
                        
                        #check POA transitions for unvoiced plosives
                        {"positive":"p d".split(),"negative":"b".split(),"true":"t".split()},
                        {"positive":"t b".split(),"negative":"d".split(),"true":"p".split()},
                        {"positive":"k d".split(),"negative":"g".split(),"true":"t".split()},  
                        
                        #check POA transitions for voiced plosives
                        {"positive":"b t".split(),"negative":"p".split(),"true":"d".split()},
                        {"positive":"d p".split(),"negative":"t".split(),"true":"b".split()},
                        {"positive":"g t".split(),"negative":"k".split(),"true":"d".split()},   
                        
                        #check POA transitions for nasals
                        {"positive":"m d".split(),"negative":"b".split(),"true":"n".split()},
                        {"positive":"n b".split(),"negative":"d".split(),"true":"m".split()},
                        {"positive":"N d".split(),"negative":"g".split(),"true":"n".split()},  
                        
                        #check POA transitions for unvoiced fricatives 
                        {"positive":"f t".split(),"negative":"p".split(),"true":"s".split()},
                        {"positive":"s p".split(),"negative":"t".split(),"true":"f".split()},
                        {"positive":"x t".split(),"negative":"k".split(),"true":"s".split()},                     
                        {"positive":"X k".split(),"negative":"q".split(),"true":"x".split()},  
                        
                        #check POA transitions for voiced fricatives 
                        {"positive":"v d".split(),"negative":"b".split(),"true":"z".split()},
                        {"positive":"z b".split(),"negative":"d".split(),"true":"v".split()},
                        
                        
                        ###################
                        #vowels
                        
                        #check for roundedness
                        {"positive":"i o".split(),"negative":"e".split(),"true":"u".split()},
                        {"positive":"e u".split(),"negative":"i".split(),"true":"o".split()},
                        
                        #check for height
                        {"positive":"i o".split(),"negative":"u".split(),"true":"o".split()},
                        {"positive":"e u".split(),"negative":"o".split(),"true":"u".split()},
                        
                        #check for +nasality 
                        {"positive":"i u*".split(),"negative":"u".split(),"true":"i*".split()},
                        {"positive":"e o*".split(),"negative":"o".split(),"true":"e*".split()},
                        {"positive":"u i*".split(),"negative":"i".split(),"true":"u*".split()},
                        {"positive":"o e*".split(),"negative":"e".split(),"true":"o*".split()},                       
                        {"positive":"a o*".split(),"negative":"o".split(),"true":"a*".split()},
                        {"positive":"E e*".split(),"negative":"e".split(),"true":"E*".split()},
                        {"positive":"3 E*".split(),"negative":"E".split(),"true":"3*".split()},
                        
                        #check for -nasality 
                        {"positive":"i* u".split(),"negative":"u*".split(),"true":"i".split()},
                        {"positive":"e* o".split(),"negative":"o*".split(),"true":"e".split()},
                        {"positive":"u* i".split(),"negative":"i*".split(),"true":"u".split()},
                        {"positive":"o* e".split(),"negative":"e*".split(),"true":"o".split()},                       
                        {"positive":"a* o".split(),"negative":"o*".split(),"true":"a".split()},
                        {"positive":"E* e".split(),"negative":"e*".split(),"true":"E".split()},
                        {"positive":"3* E".split(),"negative":"E*".split(),"true":"3".split()},

                        

                      ]
    