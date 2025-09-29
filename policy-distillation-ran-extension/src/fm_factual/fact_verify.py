# coding=utf-8
# Copyright 2023-present the International Business Machines.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Our implementation of the VeryScore paper using LLAMA3 models

import os
import json
import sys
import argparse
import torch
import litellm
import pandas as pd

from typing import List
from tqdm import tqdm
from dotenv import load_dotenv

# litellm.set_verbose = True

# Local
from fm_factual.atom_extractor import AtomExtractor
from fm_factual.atom_reviser import AtomReviser
from fm_factual.context_retriever import ContextRetriever
from fm_factual.fact_utils import Atom, Context, build_atoms, build_contexts
from fm_factual.utils import RITS_MODELS, DEFAULT_PROMPT_BEGIN, DEFAULT_PROMPT_END, extract_last_wrapped_response

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FEW_SHOTS = [
    {"claim": "Characters Lenny and Carl on The Simpsons are hearing but are depicted as close friends of the Simpsons family.", "search_result": "Search result 1\nTitle: Character Spotlight: Lenny Leonard and Carl Carlson (& Barflies)\nContent: Their friendship is a pretty singular aspect on the show -- save Bart and Milhouse (or to some degree, Mr. Burns and Smithers) -- they always ...\nLink: https://nohomers.net/forums/index.php?threads/character-spotlight-lenny-leonard-and-carl-carlson-barflies.23798/\n\nSearch result 2\nTitle: The Simpsons: Lenny and Carl's History, Explained - CBR\nContent: Introduced in the show's first season, the pair were portrayed as background characters at Homer's work, usually appearing together in minor ...\nLink: https://www.cbr.com/the-simpsons-lenny-carl-history-explained/\n\nSearch result 3\nTitle: Are Lennie and Carl Homer Simpson's real or fake friends? - Quora\nContent: Lenni is a pal, Carl doesn't consider any of them to be 'friends' they're just shallow guys he hangs out with. Lenny and Carl have a special ...\nLink: https://www.quora.com/Are-Lennie-and-Carl-Homer-Simpson-s-real-or-fake-friends\n\nSearch result 4\nTitle: [The Simpsons] Lenny and Carl aren't ambiguously gay (originally)\nContent: Theory: Lenny and Carl started out as a parody of background characters who always appear as a pair: Crabbe and Goyle or Fred and George in \" ...\nLink: https://www.reddit.com/r/FanTheories/comments/yw7bp4/the_simpsons_lenny_and_carl_arent_ambiguously_gay/\n\nSearch result 5\nTitle: Lenny Leonard | Simpsons Wiki | Fandom\nContent: He is the best friend of Carl Carlson and, along with Carl, second best friend of Homer Simpson, behind Barney Gumble and Moe Syzslak.\nLink: https://simpsons.fandom.com/wiki/Lenny_Leonard\n\nSearch result 6\nTitle: Are Simpsons' Carl & Lenny Gay? Every Clue To Their Relationship\nContent: One of The Simpsons' many mysteries is Lenny and Carl's relationship, as it has been hinted that the two are more than just best friends.\nLink: https://screenrant.com/simpsons-shows-carl-lenny-gay-couple-clues-hints/\n\nSearch result 7\nTitle: Lenny Leonard - Wikisimpsons, the Simpsons Wiki\nContent: Lenford \"Lenny\" Leonard is the best friend of Carl Carlson and, along with Carl, second best friend of Homer Simpson, behind Barney Gumble.\nLink: https://simpsonswiki.com/wiki/Lenny_Leonard\n\nSearch result 8\nTitle: Lenny | The Simpsons: Tapped Out Wiki | Fandom\nContent: Lenford \"Lenny\" Leonard MPhs (born 1960) is the best friend of Carl Carlson and, along with Carl, second best friend of Homer Simpson, behind Barney Gumble.\nLink: https://simpsonstappedout.fandom.com/wiki/Lenny\n\nSearch result 9\nTitle: The Simpsons - Wikipedia\nContent: Developed by Groening, James L. Brooks, and Sam Simon, the series is a satirical depiction of American life, epitomized by the Simpson family, which consists of ...\nLink: https://en.wikipedia.org/wiki/The_Simpsons\n\nSearch result 10\nTitle: Lenny Leonard & Carl Carlson - Friends or Couple? | The Simpsons\nContent: Embark on an Epic Friendship Adventure: Lenny Leonard and Carl Carlson from The Simpsons ...\nLink: https://www.youtube.com/watch?v=qY5hjalUhfA", "human_label": "Inconclusive"},
    {"claim": "The championship match of the FIFA World Cup 2026 will be hosted by the United States.", "search_result": "Search result 1\nTitle: World Cup 2026 | New York New Jersey to host final - FIFA\nContent: New York New Jersey Stadium has been confirmed as the location for the FIFA World Cup 26™ final on Sunday, 19 July 2026. The full match schedule for the ...\nLink: https://www.fifa.com/fifaplus/en/tournaments/mens/worldcup/canadamexicousa2026/articles/new-york-new-jersey-stadium-host-world-cup-2026-final\n\nSearch result 2\nTitle: 2026 FIFA World Cup - Wikipedia\nContent: The tournament will take place from June 11 to July 19, 2026. It will be jointly hosted by 16 cities in three North American countries: Canada, Mexico, and the ...\nLink: https://en.wikipedia.org/wiki/2026_FIFA_World_Cup\n\nSearch result 3\nTitle: World Cup 2026 | Dallas to host nine matches - FIFA\nContent: Dallas Stadium will host nine matches from the FIFA World Cup 26™, including four knockout games in the latter stages of the tournament.\nLink: https://www.fifa.com/fifaplus/en/tournaments/mens/worldcup/canadamexicousa2026/articles/dallas-stadium-host-nine-world-cup-matches\n\nSearch result 4\nTitle: When and where is the 2026 World Cup? What is the format?\nContent: 104 games will take place in the 2026 World Cup across the USA, Mexico, and Canada with 16 different host cities selected and the schedule confirmed.\nLink: https://www.nbcsports.com/soccer/news/when-and-where-is-the-2026-world-cup\n\nSearch result 5\nTitle: New York/New Jersey will host 2026 FIFA World Cup final\nContent: FIFA announced on Sunday that the final of the 2026 World Cup will be held at MetLife Stadium just outside New York.\nLink: https://www.foxsports.com/stories/soccer/new-york-new-jersey-will-host-2026-fifa-world-cup-final-mexico-city-gets-the-opener\n\nSearch result 6\nTitle: FIFA announces 2026 World Cup details, including host stadium for ...\nContent: The tournament will be hosted by the United States, Mexico and Canada. PHOTO: A general exterior view of the MetLife Stadium the home of NFL New ...\nLink: https://abcnews.go.com/Sports/new-jersey-metlife-stadium-host-2026-world-cup-final/story?id=106937005\n\nSearch result 7\nTitle: New York, New Jersey to host 2026 FIFA World Cup final - France 24\nContent: The 2026 World Cup final will be held at MetLife Stadium in New York/New Jersey, organisers FIFA announced on Sunday.\nLink: https://www.france24.com/en/americas/20240204-new-york-new-jersey-to-host-2026-fifa-world-cup-final\n\nSearch result 8\nTitle: 2026 World Cup final will be played at MetLife Stadium in New Jersey\nContent: The 2026 World Cup final will be played at MetLife Stadium in New Jersey, beating out Texas and California for soccer's showcase game.\nLink: https://www.nbcmiami.com/news/sports/miami-misses-out-on-hosting-the-2026-fifa-world-cup-final-metlife-stadium-in-new-jersey-to-host/3224690/\n\nSearch result 9\nTitle: New Jersey's MetLife Stadium to host 2026 FIFA World Cup final on ...\nContent: The 104-match tournament will open in Mexico on June 11 and will move entirely to the US from the quarterfinal round. MetLife Stadium in East ...\nLink: https://www.aljazeera.com/sports/2024/2/5/new-jersey-to-host-2026-fifa-world-cup-final\n\nSearch result 10\nTitle: World Cup 2026 | Seattle to host six matches - FIFA\nContent: Seattle Stadium will host six matches from the FIFA World Cup 26™, including USA's second group fixture and two knockout games. The full match schedule for ...\nLink: https://www.fifa.com/fifaplus/en/tournaments/mens/worldcup/canadamexicousa2026/articles/seattle-host-six-matches-stadium", "human_label": "Supported"},
    {"claim": "It is essential to understand the limitations of heating a dead battery to temporarily revive its function.", "search_result": "Search result 1\nTitle: Why do batteries come back to life if you let them rest?\nContent: By letting the battery rest, you give the reaction products a chance to dissipate. The higher the drain on the battery, the faster the products ...\nLink: https://electronics.howstuffworks.com/everyday-tech/question390.htm\n\nSearch result 2\nTitle: ELI5 Why do batteries work again after heating them up? - Reddit\nContent: The warmth heats up the battery and give it a little more juice allowing me to use it again for a minute.\nLink: https://www.reddit.com/r/explainlikeimfive/comments/vjn30b/eli5_why_do_batteries_work_again_after_heating/\n\nSearch result 3\nTitle: What happen when a dead battery and a good battery connected in ...\nContent: The batteries could get warm/hot/overheat and or get damaged depending on the capacity of the batteries being connected and their ability to ...\nLink: https://www.quora.com/What-happen-when-a-dead-battery-and-a-good-battery-connected-in-parallel\n\nSearch result 4\nTitle: The flat battery vs dead battery - key differences and how to handle t ...\nContent: A flat battery is temporarily depleted but can be recharged and regain its functionality. ... its ability to hold a charge and cannot be revived.\nLink: https://www.tycorun.com/blogs/news/flat-battery-vs-dead-battery\n\nSearch result 5\nTitle: How to Bring Back 0V/low voltage NiMH Battery to Life – EBLOfficial\nContent: In such cases, it can be essential to overcharge your NiMH battery sometimes to ensure they are fully charged and operate optimally. However, ...\nLink: https://www.eblofficial.com/blogs/blog/how-to-revive-nimh-battery\n\nSearch result 6\nTitle: How Long Should You Drive to Charge Your Car Battery?\nContent: Discover how long you should drive to charge your car battery effectively. Learn tips, calculations, and alternative methods. Drive smart, charge smart!\nLink: https://heartautocare.com/how-long-should-you-drive-to-charge-your-car-battery/\n\nSearch result 7\nTitle: BU-502: Discharging at High and Low Temperatures\nContent: Like humans, batteries function best at room temperature. Warming a dying battery in a mobile phone or flashlight in our jeans might provide ...\nLink: https://batteryuniversity.com/article/bu-502-discharging-at-high-and-low-temperatures\n\nSearch result 8\nTitle: Can You Recharge A Completely Dead Car Battery?\nContent: Learn the signs of a dead battery, whether you can recharge a completely dead car battery and other method to safely revive it.\nLink: https://carfromjapan.com/article/car-maintenance/can-you-recharge-a-completely-dead-car-battery/\n\nSearch result 9\nTitle: How to Fix a Dead Car Battery - Naylor's Auto Repair\nContent: It indicates that the battery is leaking. This will tell that your battery will not function as it should.\nLink: https://www.naylorsautorepairidaho.com/blog/how-fix-dead-car-battery\n\nSearch result 10\nTitle: The Truth About Reviving Dead Batteries - U.S. Battery Mfg. Co.\nContent: The batteries may or may not recover after one desulfation charge. If not, it may require several cycles of discharge and recharge with ...\nLink: https://www.usbattery.com/the-truth-about-reviving-dead-batteries/", "human_label": "Inconclusive"},
    {"claim": "Sarah and James were shot execution-style in the living room.", "search_result": "Search result 1\nTitle: By Taufik | There were string of armed robberies and free murders ...\nContent: handgun been in the execution-style murder but we ... quickly cleared the living room, went ...\nLink: https://www.facebook.com/Haru6789.cv/videos/watch-killer-siblings-season-3-episode-5-allridges/411826944038089/\n\nSearch result 2\nTitle: Why are Austin's Yogurt Shop Murders Still Unsolved After 30 Years?\nContent: Thomas, Ayers, and Jennifer and Sarah Harbinson were all shot and killed execution-style with a single gunshot wound to the back of their heads.\nLink: https://www.oxygen.com/crime-news/yogurt-shop-murders-austin-solved-theories-dna-update\n\nSearch result 3\nTitle: Jenean Chapman's neighbors say they heard former aide to Sarah ...\nContent: Neighbors revealed they heard 'screaming fights' between Sarah Ferguson's ex-assistant and her husband, James Patrick, who has been arrested ...\nLink: https://www.dailymail.co.uk/news/article-12582157/Neighbors-Sarah-Ferguson-fights-James-Patrick-Jenean-Chapman-Dallas-Texas.html\n\nSearch result 4\nTitle: Sarah Brady, gun control advocate and widow of James Brady, dies ...\nContent: Sarah Brady, gun control advocate and widow of James Brady, dies at 73 ... After the 1981 assassination attempt on President Ronald Reagan, nearly ...\nLink: https://www.latimes.com/local/obituaries/la-me-sarah-brady-20150403-story.html\n\nSearch result 5\nTitle: Slayings of 3 Youths Stun Little Rock - The New York Times\nContent: When three armed men invaded the home of Mary Hussian here this month and killed three of her children, execution-style, this city, ...\nLink: https://www.nytimes.com/1995/06/27/us/slayings-of-3-youths-stun-little-rock.html\n\nSearch result 6\nTitle: Revenge Is Motive in 7 Bronx Slayings, Police Say - The New York ...\nContent: Valentine's Day, they said, when three adults and three teen-agers were lined up on the living-room floor of a Bronx apartment and each shot in ...\nLink: https://www.nytimes.com/1993/02/26/nyregion/revenge-is-motive-in-7-bronx-slayings-police-say.html\n\nSearch result 7\nTitle: 20 years later, triple murder casts a shadow over Christmas ...\nContent: The grisly, Christmas Day discovery of a Jean Wholaver and her two daughters shot dead execution-style by their ex-husband and father, Ernest ...\nLink: https://www.pennlive.com/crime/2022/12/20-years-later-triple-murder-casts-a-shadow-over-christmas-memories-in-middletown.html\n\nSearch result 8\nTitle: Murder victims identified after police say they were shot 'execution ...\nContent: Murder victims identified after police say they were shot 'execution style'. 7.6K views · 1 ...\nLink: https://www.youtube.com/watch?v=ux9L1NLm-BI\n\nSearch result 9\nTitle: Bonus – the James Bigby murders - what was that like\nContent: The former auto mechanic is accused of shooting three former coworkers and strangling a 4-month-old baby. We have more in this NBC report. The 4 ...\nLink: https://whatwasthatlike.com/2023/04/14/bonus-the-james-bigby-murders/\n\nSearch result 10\nTitle: Man arrested over 'execution-style' killings in Philadelphia park\nContent: A man has been arrested over the \"execution-style\" killings of a man and woman in a Philadelphia park. The bodies of Thurston Cooper, 49, ...\nLink: https://www.irishstar.com/news/pennsylvania-news/execution-style-killings-philadelphia-park-32278732", "human_label": "Inconclusive"},
    {"claim": "Vikings used their longships to transport livestock.", "search_result": "Search result 1\nTitle: How did the Vikings transport animals on their ships? - Quora\nContent: The Vikings transported horses overseas in boats very similar to Viking longships, but with flat flooring built within the hulls, which allowed ...\nLink: https://www.quora.com/How-did-the-Vikings-transport-animals-on-their-ships\n\nSearch result 2\nTitle: The Truth Behind Vikings Ships\nContent: They could land on any beach, permitting lightning-quick embarking and attacks. Great loads could be carried, including horses and livestock.\nLink: https://www.vikings.com/news/the-truth-behind-vikings-ships-18274806\n\nSearch result 3\nTitle: Viking ships | Royal Museums Greenwich\nContent: Cargo vessels were used to carry trade goods and possessions. They were wider than the longships and travelled more slowly.\nLink: https://www.rmg.co.uk/stories/topics/viking-ships\n\nSearch result 4\nTitle: How did the vikings bring the horses to Iceland? - Reddit\nContent: This was stored in casks and skins. It's worth mentioning that some discomfort for these animals was expected, and they may have been 'rationed' ...\nLink: https://www.reddit.com/r/AskHistorians/comments/5l05tj/how_did_the_vikings_bring_the_horses_to_iceland/\n\nSearch result 5\nTitle: Did the Vikings bring livestock on their raids and voyages?\nContent: The Vikings did not generally transport livestock during their raids. Viking longships did not have substantial cargo room. As a result, the Vikings only ...\nLink: https://homework.study.com/explanation/did-the-vikings-bring-livestock-on-their-raids-and-voyages.html\n\nSearch result 6\nTitle: LEVS : Viking FAQs : Transportation\nContent: The upper deck was used for humans and their space, and the lower decks were used for livestock, furniture, and other goods that were being transported from ...\nLink: http://vikingship.org/ourfaqs/transportation_1.html\n\nSearch result 7\nTitle: Viking Ships - World History Encyclopedia\nContent: Such ships gave the Vikings the ability to trade, make war, carry animals ... Longships, on the other hand, used both oars and sails to reach ...\nLink: https://www.worldhistory.org/Viking_Ships/\n\nSearch result 8\nTitle: The Viking Longship - Warfare History Network\nContent: Longships enabled the Vikings to transport their armies throughout Europe and conduct amphibious assaults in estuaries and navigable rivers.\nLink: https://warfarehistorynetwork.com/article/the-viking-longship/\n\nSearch result 9\nTitle: Viking Ships | Regia Anglorum\nContent: Most Viking trading boats were able to come ashore without the need of complicated harbour facilities, and yet carry many tons of cargo. This encouraged ...\nLink: https://regia.org/research/ships/Ships0.htm", "human_label": "Contradicted"},
    {"claim": "Romário has scored a total of 92 international goals.", "search_result": "Search result 1\nTitle: Romário - Wikipedia\nContent: A prolific striker renowned for his clinical finishing, he scored over 700 goals and is one of the few players to score at least 100 goals for three clubs. He ...\nLink: https://en.wikipedia.org/wiki/Rom%C3%A1rio\n\nSearch result 2\nTitle: Romário de Souza Faria - Goals in International Matches - RSSSF\nContent: The outstanding Brazilian player, who scored 55 international goals for his National Team, has ... 92 Paris France 2-0 27 10 16/12/92 Porto ...\nLink: https://www.rsssf.org/miscellaneous/romario-intlg.html\n\nSearch result 3\nTitle: In 92 Romário said:\" I didn't come from Europe for a friendly in Brazil ...\nContent: he played till he was like 42, scored 772 goals in official games in his career (the same average as messi and higher than ronaldo) and won a ...\nLink: https://www.reddit.com/r/soccer/comments/7klck5/in_92_rom%C3%A1rio_said_i_didnt_come_from_europe_for_a/\n\nSearch result 4\nTitle: Romário - National team - Transfermarkt\nContent: Only games with goals scored. Compact · Detailed. Matchday, Date, Match, Pos. International Friendlies. 5/23/87, Republic of Ireland · 1:0 ...\nLink: https://www.transfermarkt.us/romario/nationalmannschaft/spieler/7942\n\nSearch result 5\nTitle: List of men's footballers with 50 or more international goals - Wikipedia\nContent: Cristiano Ronaldo holds the all-time record with 128 goals. Brazil and Hungary hold the record of having the most players to have scored 50 or more ...\nLink: https://en.wikipedia.org/wiki/List_of_men%27s_footballers_with_50_or_more_international_goals\n\nSearch result 6\nTitle: romário de souza faria - Romàrio - FC Barcelona Players\nContent: According to recent studies, he is the highest goal scorer in world football, with 768 goals in official matches for club and national team, ahead of Bican, ...\nLink: https://players.fcbarcelona.com/en/player/759-romario-romario-souza-faria\n\nSearch result 7\nTitle: Romario is the best striker ever imo | BigSoccer Forum\nContent: * Dutch league's top scorer: 1989, 1990, 1991, 1992 ... romario has scored 55 goals , thats about 34 % of ... in 1994 world cup, in brazil games was scored a total ...\nLink: https://www.bigsoccer.com/threads/romario-is-the-best-striker-ever-imo.597679/\n\nSearch result 8\nTitle: Romario: The Art of Goal-Scoring Brilliance. - LinkedIn\nContent: In 1997 alone, the duo scored an impressive total of 34 international goals with 19 coming from Romário. The Ro-Ro attack was expected to ...\nLink: https://www.linkedin.com/pulse/romario-art-goal-scoring-brilliance-javad-ghorbani\n\nSearch result 9\nTitle: Who is Brazil's leading all-time top goal scorer? Pele, Neymar ...\nContent: The three-time World Cup winner went on to score a total of 77 goals in 92 games for the Selecao at an astonishing 0.84 goals-per-game ratio.\nLink: https://www.goal.com/en-us/lists/pele-neymar-ronaldo-who-is-brazils-leading-all-time-top-goalscorers/blt75e4aa20fd6a2bed\n\nSearch result 10\nTitle: Romário - All goals - Transfermarkt\nContent: This overview shows all goals scored by the selected player, including the results of the games. It is also possible to select a competition. Filter by season:.\nLink: https://www.transfermarkt.us/romario/alletore/spieler/7942", "human_label": "Contradicted"},
    {"claim": "Utopia portrays a society that values education and learning.", "search_result": "Search result 1\nTitle: Utopia Education, Science, Philosophy Summary & Analysis\nContent: The Utopians believe that it is through education that the values and dispositions of citizens are molded. The success of the Utopian educational system is ...\nLink: https://www.sparknotes.com/philosophy/utopia/section10/\n\nSearch result 2\nTitle: Utopianism and Education: The Legacy of Thomas More - jstor\nContent: Utopia, are filled from this sector of society. ... fail to find significance and meaning in classroom learning. ... closed, value positions about education - about ...\nLink: https://www.jstor.org/stable/3122242\n\nSearch result 3\nTitle: [PDF] education in thomas more's utopia - DergiPark\nContent: This work of art depicts an ideal society and in doing so relays many qualities desirable in an educational system. This study aims to explore Thomas More's ...\nLink: https://dergipark.org.tr/tr/download/article-file/55667\n\nSearch result 4\nTitle: Society and Government in Thomas More's “Utopia” - Medium\nContent: The utopian society portrayed in More's book employs enslaved people for animal slaughter and heavy labour. However, the discussion of slavery ...\nLink: https://medium.com/@batuhankarakus95/society-and-government-in-thomas-mores-utopia-a-focus-on-values-and-functioning-d9626aeaee71\n\nSearch result 5\nTitle: Utopia by Thomas More | Summary, Characters & Themes - Study.com\nContent: According to Thomas More, Utopian society is based on rational thought, communal property, productivity, no class distinctions or poverty, little crime or ...\nLink: https://study.com/academy/lesson/utopia-by-thomas-more-summary-analysis-quiz.html\n\nSearch result 6\nTitle: Thomas More's Utopian Education | Inside Classical Education\nContent: We keep hearing of Utopian visions of culture and society, and I have been itching to go to the sources the word and the concept. More's book ...\nLink: https://insideclassicaled.com/thomas-mores-utopian-education/\n\nSearch result 7\nTitle: About Utopia and Utopian Literature - Cliffs Notes\nContent: Throughout the society, life is directed by a highly moral code of conduct. An educational system for the intelligentsia is elaborately and idealistically ...\nLink: https://www.cliffsnotes.com/literature/u/utopia-utopian-literature/about-utopia-and-utopian-literature\n\nSearch result 8\nTitle: Full article: Educational Studies and the Domestication of Utopia\nContent: Utopia depicts an entire functioning society. Utopian visions 'are explicitly holistic, imaginary, critical, normative, prescriptive ...\nLink: https://www.tandfonline.com/doi/full/10.1080/00071005.2016.1143085\n\nSearch result 9\nTitle: Utopian Society | Definition, Ideas & Examples - Lesson - Study.com\nContent: The idea behind utopianism is a society in which everyone's needs are met and society's ills have been defeated. Because this is an extremely tall order, ...\nLink: https://study.com/academy/lesson/characteristics-of-a-utopian-society.html\n\nSearch result 10\nTitle: Education and Utopia: Robert Owen and Charles Fourier - Jstor\nContent: into the community—his ideal society contains no schools and no teachers. ... (Universities are not mentioned in A new view of society, but are later portrayed as ...\nLink: https://www.jstor.org/stable/23119459", "human_label": "Supported"},
    {"claim": "The higher density of water can cause sound waves to be reflected or refracted differently.", "search_result": "Search result 1\nTitle: How does sound in air differ from sound in water?\nContent: Sounds in water and sounds in air that have the same pressures have very different intensities because the density of water is much greater than ...\nLink: https://dosits.org/science/sounds-in-the-sea/how-does-sound-in-air-differ-from-sound-in-water/\n\nSearch result 2\nTitle: When a sound wave passes from air into water, what properties of ...\nContent: Sound travels faster in water than in air because the density of water is higher. The exact speed depends on the temperature, pressure, and ...\nLink: https://www.quora.com/When-a-sound-wave-passes-from-air-into-water-what-properties-of-the-wave-will-change-1\n\nSearch result 3\nTitle: Reflection, Refraction, and Diffraction - The Physics Classroom\nContent: Sound waves travel slower in cooler air than they do in warmer air. For this reason, the portion of the wavefront directly above the water is slowed down, while ...\nLink: https://www.physicsclassroom.com/class/sound/Lesson-3/Reflection,-Refraction,-and-Diffraction\n\nSearch result 4\nTitle: Refraction of Sound Waves - Graduate Program in Acoustics\nContent: When a wave encounters different medium where the wave speed is different, the wave will change directions. Most often refraction is encountered in a study ...\nLink: https://www.acs.psu.edu/drussell/demos/refract/refract.html\n\nSearch result 5\nTitle: Refraction of light - Science Learning Hub\nContent: When light travels from air into water, it slows down, causing it to change direction slightly. This change of direction is called refraction.\nLink: https://www.sciencelearn.org.nz/resources/49-refraction-of-light\n\nSearch result 6\nTitle: How is sound refracted going from a less dense media to a denser ...\nContent: Refraction is the property of a wave to bend as it propagates through different media. The changes in the media makes one side of the wave slow down or speed up ...\nLink: https://homework.study.com/explanation/how-is-sound-refracted-going-from-a-less-dense-media-to-a-denser-media.html\n\nSearch result 7\nTitle: Reflection, Refraction, and Diffraction - The Physics Classroom\nContent: Diffraction of water waves is observed in a harbor as waves bend around small boats and are found to disturb the water behind them. The same waves however are ...\nLink: https://www.physicsclassroom.com/class/waves/Lesson-3/Reflection,-Refraction,-and-Diffraction\n\nSearch result 8\nTitle: What happens to a sound wave as it travels from air into water? (a ...\nContent: The sound intensity in the water will be less than it was in air because some sound is reflected by the water surface. However, the frequency ( ...\nLink: https://www.toppr.com/ask/question/what-happens-to-a-sound-wave-as-it-travels-from-air-into-water-a-its/\n\nSearch result 9\nTitle: Ultrasound Physics and Instrumentation - StatPearls - NCBI Bookshelf\nContent: The difference in structure density promotes the refraction or bending of sound waves off the surface. The result is that echoes do not return ...\nLink: https://www.ncbi.nlm.nih.gov/books/NBK570593/\n\nSearch result 10\nTitle: How does sound propagate from air into water?\nContent: A portion of the sound wave will reflect away from the water and into the air, while another part will transmit into the water. During ...\nLink: https://dosits.org/science/movement/how-does-sound-propagate-from-air-into-water/", "human_label": "Supported"},
    {"claim": "Mount Katahdin is 6,288.2 feet (1,917.6 meters) tall.", "search_result": "Search result 1\nTitle: Mount Katahdin - Wikipedia\nContent: Mount Katahdin is the highest mountain in the U.S. state of Maine at 5,269 feet (1,606 m). Named Katahdin, which means \"Great Mountain\", by the Penobscot ...\nLink: https://en.wikipedia.org/wiki/Mount_Katahdin\n\nSearch result 2\nTitle: Mount Katahdin - Baxter State Park (U.S. National Park Service)\nContent: Katahdin, which translates to \"Greatest Mountain\" in Penobscot, is the highest mountain in the state of Maine at 5,269 feet.\nLink: https://www.nps.gov/places/katahdin-baxter-state-park.htm\n\nSearch result 3\nTitle: Mount Katahdin | Maine, Map, & Facts | Britannica\nContent: Mount Katahdin, highest point (5268 feet [1606 metres]) in Maine, U.S. It lies in Baxter State Park, 20 miles (32 km) northwest of Millinocket, ...\nLink: https://www.britannica.com/place/Mount-Katahdin\n\nSearch result 4\nTitle: Mount Katahdin - Simple English Wikipedia, the free encyclopedia\nContent: It is 5,267 feet (1,605 m) tall. Mount Katahdin. Katahdin from 10,000 ft (3,000 m). Highest point. Elevation, 5,267 ft (1,605 m)NAVD 88 · Prominence, 4,288 ft ( ...\nLink: https://simple.wikipedia.org/wiki/Mount_Katahdin\n\nSearch result 5\nTitle: Mount Katahdin- The Beginning - Alt Route Meals\nContent: It is the states highest peak standing at 5,269ft tall. This is the Northern Terminus of the Appalachian Trail, a 2,189 mile footpath that ...\nLink: https://altroutemeals.com/blogs/news/thai-curry-maine-katahdin-millinocket\n\nSearch result 6\nTitle: Mount Katahdin - PeakVisor\nContent: Mount Katahdin is the highest mountain in the U.S. state of Maine at 5,267 feet (1,605 m). Named Katahdin by the Penobscot Indians, which means \"The ...\nLink: https://peakvisor.com/peak/mount-katahdin.html\n\nSearch result 7\nTitle: Katahdin, Maine - Peakbagger.com\nContent: Elevation: 5268 feet, 1606 meters ; Highest Summit, Baxter Peak ; Subpeaks, Katahdin - South Peak (5260 ft/1603 m) Pamola Peak (4919 ft/1499 m) Chimney Peak (4900 ...\nLink: https://peakbagger.com/Peak.aspx?pid=6820\n\nSearch result 8\nTitle: Summit to Mt. Katahdin - Tallest Peak in Maine. - YouTube\nContent: In 1989, I attempted to summit Mt. Katahdin as a Boy Scout while attending a High Adventure ...\nLink: https://www.youtube.com/watch?v=t-bT2clu57o", "human_label": "Contradicted"}
]

VERISCORE_PROMPT = """{_PROMPT_BEGIN_PLACEHOLDER}

You need to judge whether a claim is supported or contradicted by Google search results, or whether there is no enough information to make the judgement. When doing the task, take into consideration whether the link of the search result is of a trustworthy source. Mark your answer with ### signs.

Below are the definitions of the three categories:

Supported: A claim is supported by the search results if everything in the claim is supported and nothing is contradicted by the search results. There can be some search results that are not fully related to the claim.
Contradicted: A claim is contradicted by the search results if something in the claim is contradicted by some search results. There should be no search result that supports the same part.
Inconclusive: A claim is inconclusive based on the search results if:
- a part of a claim cannot be verified by the search results,
- a part of a claim is supported and contradicted by different pieces of evidence,
- the entity/person mentioned in the claim has no clear referent (e.g., "the approach", "Emily", "a book").

Here are some examples:

Claim: {}

{}

Your decision: ###{}###

Claim: {}

{}

Your decision: ###{}###

Claim: {}

{}

Your decision: ###{}###

Claim: {}

{}

Your decision: ###{}###

Claim: {}

{}

Your decision: ###{}###

Claim: {}

{}

Your decision: ###{}###

Claim: {}

{}

Your decision: ###{}###

Claim: {}

{}

Your decision: ###{}###

Claim: {}

{}

Your decision: ###{}###

Your task:
Claim: {_CLAIM_PLACEHOLDER}

{_SEARCH_RESULTS_PLACEHOLDER}

Your decision:{_PROMPT_END_PLACEHOLDER}
"""

FEW_SHOTS_SHORT = [
    {"claim": "Characters Lenny and Carl on The Simpsons are hearing but are depicted as close friends of the Simpsons family.", "search_result": "Search result 1\nTitle: Character Spotlight: Lenny Leonard and Carl Carlson (& Barflies)\nContent: Their friendship is a pretty singular aspect on the show -- save Bart and Milhouse (or to some degree, Mr. Burns and Smithers) -- they always ...\nLink: https://nohomers.net/forums/index.php?threads/character-spotlight-lenny-leonard-and-carl-carlson-barflies.23798/\n\nSearch result 2\nTitle: The Simpsons: Lenny and Carl's History, Explained - CBR\nContent: Introduced in the show's first season, the pair were portrayed as background characters at Homer's work, usually appearing together in minor ...\nLink: https://www.cbr.com/the-simpsons-lenny-carl-history-explained/\n\nSearch result 3\nTitle: Are Lennie and Carl Homer Simpson's real or fake friends? - Quora\nContent: Lenni is a pal, Carl doesn't consider any of them to be 'friends' they're just shallow guys he hangs out with. Lenny and Carl have a special ...\nLink: https://www.quora.com/Are-Lennie-and-Carl-Homer-Simpson-s-real-or-fake-friends\n\nSearch result 4\nTitle: [The Simpsons] Lenny and Carl aren't ambiguously gay (originally)\nContent: Theory: Lenny and Carl started out as a parody of background characters who always appear as a pair: Crabbe and Goyle or Fred and George in \" ...\nLink: https://www.reddit.com/r/FanTheories/comments/yw7bp4/the_simpsons_lenny_and_carl_arent_ambiguously_gay/\n\nSearch result 5\nTitle: Lenny Leonard | Simpsons Wiki | Fandom\nContent: He is the best friend of Carl Carlson and, along with Carl, second best friend of Homer Simpson, behind Barney Gumble and Moe Syzslak.\nLink: https://simpsons.fandom.com/wiki/Lenny_Leonard", "human_label": "Inconclusive"},
    {"claim": "The championship match of the FIFA World Cup 2026 will be hosted by the United States.", "search_result": "Search result 1\nTitle: World Cup 2026 | New York New Jersey to host final - FIFA\nContent: New York New Jersey Stadium has been confirmed as the location for the FIFA World Cup 26™ final on Sunday, 19 July 2026. The full match schedule for the ...\nLink: https://www.fifa.com/fifaplus/en/tournaments/mens/worldcup/canadamexicousa2026/articles/new-york-new-jersey-stadium-host-world-cup-2026-final\n\nSearch result 2\nTitle: 2026 FIFA World Cup - Wikipedia\nContent: The tournament will take place from June 11 to July 19, 2026. It will be jointly hosted by 16 cities in three North American countries: Canada, Mexico, and the ...\nLink: https://en.wikipedia.org/wiki/2026_FIFA_World_Cup\n\nSearch result 3\nTitle: World Cup 2026 | Dallas to host nine matches - FIFA\nContent: Dallas Stadium will host nine matches from the FIFA World Cup 26™, including four knockout games in the latter stages of the tournament.\nLink: https://www.fifa.com/fifaplus/en/tournaments/mens/worldcup/canadamexicousa2026/articles/dallas-stadium-host-nine-world-cup-matches\n\nSearch result 4\nTitle: When and where is the 2026 World Cup? What is the format?\nContent: 104 games will take place in the 2026 World Cup across the USA, Mexico, and Canada with 16 different host cities selected and the schedule confirmed.\nLink: https://www.nbcsports.com/soccer/news/when-and-where-is-the-2026-world-cup\n\nSearch result 5\nTitle: New York/New Jersey will host 2026 FIFA World Cup final\nContent: FIFA announced on Sunday that the final of the 2026 World Cup will be held at MetLife Stadium just outside New York.\nLink: https://www.foxsports.com/stories/soccer/new-york-new-jersey-will-host-2026-fifa-world-cup-final-mexico-city-gets-the-opener", "human_label": "Supported"},
    {"claim": "It is essential to understand the limitations of heating a dead battery to temporarily revive its function.", "search_result": "Search result 1\nTitle: Why do batteries come back to life if you let them rest?\nContent: By letting the battery rest, you give the reaction products a chance to dissipate. The higher the drain on the battery, the faster the products ...\nLink: https://electronics.howstuffworks.com/everyday-tech/question390.htm\n\nSearch result 2\nTitle: ELI5 Why do batteries work again after heating them up? - Reddit\nContent: The warmth heats up the battery and give it a little more juice allowing me to use it again for a minute.\nLink: https://www.reddit.com/r/explainlikeimfive/comments/vjn30b/eli5_why_do_batteries_work_again_after_heating/\n\nSearch result 3\nTitle: What happen when a dead battery and a good battery connected in ...\nContent: The batteries could get warm/hot/overheat and or get damaged depending on the capacity of the batteries being connected and their ability to ...\nLink: https://www.quora.com/What-happen-when-a-dead-battery-and-a-good-battery-connected-in-parallel\n\nSearch result 4\nTitle: The flat battery vs dead battery - key differences and how to handle t ...\nContent: A flat battery is temporarily depleted but can be recharged and regain its functionality. ... its ability to hold a charge and cannot be revived.\nLink: https://www.tycorun.com/blogs/news/flat-battery-vs-dead-battery\n\nSearch result 5\nTitle: How to Bring Back 0V/low voltage NiMH Battery to Life – EBLOfficial\nContent: In such cases, it can be essential to overcharge your NiMH battery sometimes to ensure they are fully charged and operate optimally. However, ...\nLink: https://www.eblofficial.com/blogs/blog/how-to-revive-nimh-battery", "human_label": "Inconclusive"},
    {"claim": "Sarah and James were shot execution-style in the living room.", "search_result": "Search result 1\nTitle: By Taufik | There were string of armed robberies and free murders ...\nContent: handgun been in the execution-style murder but we ... quickly cleared the living room, went ...\nLink: https://www.facebook.com/Haru6789.cv/videos/watch-killer-siblings-season-3-episode-5-allridges/411826944038089/\n\nSearch result 2\nTitle: Why are Austin's Yogurt Shop Murders Still Unsolved After 30 Years?\nContent: Thomas, Ayers, and Jennifer and Sarah Harbinson were all shot and killed execution-style with a single gunshot wound to the back of their heads.\nLink: https://www.oxygen.com/crime-news/yogurt-shop-murders-austin-solved-theories-dna-update\n\nSearch result 3\nTitle: Jenean Chapman's neighbors say they heard former aide to Sarah ...\nContent: Neighbors revealed they heard 'screaming fights' between Sarah Ferguson's ex-assistant and her husband, James Patrick, who has been arrested ...\nLink: https://www.dailymail.co.uk/news/article-12582157/Neighbors-Sarah-Ferguson-fights-James-Patrick-Jenean-Chapman-Dallas-Texas.html\n\nSearch result 4\nTitle: Sarah Brady, gun control advocate and widow of James Brady, dies ...\nContent: Sarah Brady, gun control advocate and widow of James Brady, dies at 73 ... After the 1981 assassination attempt on President Ronald Reagan, nearly ...\nLink: https://www.latimes.com/local/obituaries/la-me-sarah-brady-20150403-story.html\n\nSearch result 5\nTitle: Slayings of 3 Youths Stun Little Rock - The New York Times\nContent: When three armed men invaded the home of Mary Hussian here this month and killed three of her children, execution-style, this city, ...\nLink: https://www.nytimes.com/1995/06/27/us/slayings-of-3-youths-stun-little-rock.html", "human_label": "Inconclusive"},
    {"claim": "Vikings used their longships to transport livestock.", "search_result": "Search result 1\nTitle: How did the Vikings transport animals on their ships? - Quora\nContent: The Vikings transported horses overseas in boats very similar to Viking longships, but with flat flooring built within the hulls, which allowed ...\nLink: https://www.quora.com/How-did-the-Vikings-transport-animals-on-their-ships\n\nSearch result 2\nTitle: The Truth Behind Vikings Ships\nContent: They could land on any beach, permitting lightning-quick embarking and attacks. Great loads could be carried, including horses and livestock.\nLink: https://www.vikings.com/news/the-truth-behind-vikings-ships-18274806\n\nSearch result 3\nTitle: Viking ships | Royal Museums Greenwich\nContent: Cargo vessels were used to carry trade goods and possessions. They were wider than the longships and travelled more slowly.\nLink: https://www.rmg.co.uk/stories/topics/viking-ships\n\nSearch result 4\nTitle: How did the vikings bring the horses to Iceland? - Reddit\nContent: This was stored in casks and skins. It's worth mentioning that some discomfort for these animals was expected, and they may have been 'rationed' ...\nLink: https://www.reddit.com/r/AskHistorians/comments/5l05tj/how_did_the_vikings_bring_the_horses_to_iceland/\n\nSearch result 5\nTitle: Did the Vikings bring livestock on their raids and voyages?\nContent: The Vikings did not generally transport livestock during their raids. Viking longships did not have substantial cargo room. As a result, the Vikings only ...\nLink: https://homework.study.com/explanation/did-the-vikings-bring-livestock-on-their-raids-and-voyages.html", "human_label": "Contradicted"},
    # {"claim": "Romário has scored a total of 92 international goals.", "search_result": "Search result 1\nTitle: Romário - Wikipedia\nContent: A prolific striker renowned for his clinical finishing, he scored over 700 goals and is one of the few players to score at least 100 goals for three clubs. He ...\nLink: https://en.wikipedia.org/wiki/Rom%C3%A1rio\n\nSearch result 2\nTitle: Romário de Souza Faria - Goals in International Matches - RSSSF\nContent: The outstanding Brazilian player, who scored 55 international goals for his National Team, has ... 92 Paris France 2-0 27 10 16/12/92 Porto ...\nLink: https://www.rsssf.org/miscellaneous/romario-intlg.html\n\nSearch result 3\nTitle: In 92 Romário said:\" I didn't come from Europe for a friendly in Brazil ...\nContent: he played till he was like 42, scored 772 goals in official games in his career (the same average as messi and higher than ronaldo) and won a ...\nLink: https://www.reddit.com/r/soccer/comments/7klck5/in_92_rom%C3%A1rio_said_i_didnt_come_from_europe_for_a/\n\nSearch result 4\nTitle: Romário - National team - Transfermarkt\nContent: Only games with goals scored. Compact · Detailed. Matchday, Date, Match, Pos. International Friendlies. 5/23/87, Republic of Ireland · 1:0 ...\nLink: https://www.transfermarkt.us/romario/nationalmannschaft/spieler/7942\n\nSearch result 5\nTitle: List of men's footballers with 50 or more international goals - Wikipedia\nContent: Cristiano Ronaldo holds the all-time record with 128 goals. Brazil and Hungary hold the record of having the most players to have scored 50 or more ...\nLink: https://en.wikipedia.org/wiki/List_of_men%27s_footballers_with_50_or_more_international_goals", "human_label": "Contradicted"},
    # {"claim": "Utopia portrays a society that values education and learning.", "search_result": "Search result 1\nTitle: Utopia Education, Science, Philosophy Summary & Analysis\nContent: The Utopians believe that it is through education that the values and dispositions of citizens are molded. The success of the Utopian educational system is ...\nLink: https://www.sparknotes.com/philosophy/utopia/section10/\n\nSearch result 2\nTitle: Utopianism and Education: The Legacy of Thomas More - jstor\nContent: Utopia, are filled from this sector of society. ... fail to find significance and meaning in classroom learning. ... closed, value positions about education - about ...\nLink: https://www.jstor.org/stable/3122242\n\nSearch result 3\nTitle: [PDF] education in thomas more's utopia - DergiPark\nContent: This work of art depicts an ideal society and in doing so relays many qualities desirable in an educational system. This study aims to explore Thomas More's ...\nLink: https://dergipark.org.tr/tr/download/article-file/55667\n\nSearch result 4\nTitle: Society and Government in Thomas More's “Utopia” - Medium\nContent: The utopian society portrayed in More's book employs enslaved people for animal slaughter and heavy labour. However, the discussion of slavery ...\nLink: https://medium.com/@batuhankarakus95/society-and-government-in-thomas-mores-utopia-a-focus-on-values-and-functioning-d9626aeaee71\n\nSearch result 5\nTitle: Utopia by Thomas More | Summary, Characters & Themes - Study.com\nContent: According to Thomas More, Utopian society is based on rational thought, communal property, productivity, no class distinctions or poverty, little crime or ...\nLink: https://study.com/academy/lesson/utopia-by-thomas-more-summary-analysis-quiz.html", "human_label": "Supported"},
    # {"claim": "The higher density of water can cause sound waves to be reflected or refracted differently.", "search_result": "Search result 1\nTitle: How does sound in air differ from sound in water?\nContent: Sounds in water and sounds in air that have the same pressures have very different intensities because the density of water is much greater than ...\nLink: https://dosits.org/science/sounds-in-the-sea/how-does-sound-in-air-differ-from-sound-in-water/\n\nSearch result 2\nTitle: When a sound wave passes from air into water, what properties of ...\nContent: Sound travels faster in water than in air because the density of water is higher. The exact speed depends on the temperature, pressure, and ...\nLink: https://www.quora.com/When-a-sound-wave-passes-from-air-into-water-what-properties-of-the-wave-will-change-1\n\nSearch result 3\nTitle: Reflection, Refraction, and Diffraction - The Physics Classroom\nContent: Sound waves travel slower in cooler air than they do in warmer air. For this reason, the portion of the wavefront directly above the water is slowed down, while ...\nLink: https://www.physicsclassroom.com/class/sound/Lesson-3/Reflection,-Refraction,-and-Diffraction\n\nSearch result 4\nTitle: Refraction of Sound Waves - Graduate Program in Acoustics\nContent: When a wave encounters different medium where the wave speed is different, the wave will change directions. Most often refraction is encountered in a study ...\nLink: https://www.acs.psu.edu/drussell/demos/refract/refract.html\n\nSearch result 5\nTitle: Refraction of light - Science Learning Hub\nContent: When light travels from air into water, it slows down, causing it to change direction slightly. This change of direction is called refraction.\nLink: https://www.sciencelearn.org.nz/resources/49-refraction-of-light", "human_label": "Supported"},
    # {"claim": "Mount Katahdin is 6,288.2 feet (1,917.6 meters) tall.", "search_result": "Search result 1\nTitle: Mount Katahdin - Wikipedia\nContent: Mount Katahdin is the highest mountain in the U.S. state of Maine at 5,269 feet (1,606 m). Named Katahdin, which means \"Great Mountain\", by the Penobscot ...\nLink: https://en.wikipedia.org/wiki/Mount_Katahdin\n\nSearch result 2\nTitle: Mount Katahdin - Baxter State Park (U.S. National Park Service)\nContent: Katahdin, which translates to \"Greatest Mountain\" in Penobscot, is the highest mountain in the state of Maine at 5,269 feet.\nLink: https://www.nps.gov/places/katahdin-baxter-state-park.htm\n\nSearch result 3\nTitle: Mount Katahdin | Maine, Map, & Facts | Britannica\nContent: Mount Katahdin, highest point (5268 feet [1606 metres]) in Maine, U.S. It lies in Baxter State Park, 20 miles (32 km) northwest of Millinocket, ...\nLink: https://www.britannica.com/place/Mount-Katahdin\n\nSearch result 4\nTitle: Mount Katahdin - Simple English Wikipedia, the free encyclopedia\nContent: It is 5,267 feet (1,605 m) tall. Mount Katahdin. Katahdin from 10,000 ft (3,000 m). Highest point. Elevation, 5,267 ft (1,605 m)NAVD 88 · Prominence, 4,288 ft ( ...\nLink: https://simple.wikipedia.org/wiki/Mount_Katahdin\n\nSearch result 5\nTitle: Mount Katahdin- The Beginning - Alt Route Meals\nContent: It is the states highest peak standing at 5,269ft tall. This is the Northern Terminus of the Appalachian Trail, a 2,189 mile footpath that ...\nLink: https://altroutemeals.com/blogs/news/thai-curry-maine-katahdin-millinocket", "human_label": "Contradicted"}
]

VERISCORE_PROMPT_SHORT = """{_PROMPT_BEGIN_PLACEHOLDER}

You need to judge whether a claim is supported or contradicted by Google search results, or whether there is no enough information to make the judgement. When doing the task, take into consideration whether the link of the search result is of a trustworthy source. Mark your answer with ### signs.

Below are the definitions of the three categories:

Supported: A claim is supported by the search results if everything in the claim is supported and nothing is contradicted by the search results. There can be some search results that are not fully related to the claim.
Contradicted: A claim is contradicted by the search results if something in the claim is contradicted by some search results. There should be no search result that supports the same part.
Inconclusive: A claim is inconclusive based on the search results if:
- a part of a claim cannot be verified by the search results,
- a part of a claim is supported and contradicted by different pieces of evidence,
- the entity/person mentioned in the claim has no clear referent (e.g., "the approach", "Emily", "a book").

Here are some examples:

Claim: {}

{}

Your decision: ###{}###

Claim: {}

{}

Your decision: ###{}###

Claim: {}

{}

Your decision: ###{}###

Claim: {}

{}

Your decision: ###{}###

Claim: {}

{}

Your decision: ###{}###

Your task:
Claim: {_CLAIM_PLACEHOLDER}

{_SEARCH_RESULTS_PLACEHOLDER}

Your decision:{_PROMPT_END_PLACEHOLDER}
"""


class FactVerify:
    """
    Our implementation of the VeriScore paper. Atomic units are verified using
    Google search results only (without retrieving the context i.e., text).
    """

    def __init__(
            self,
            context_retriever: ContextRetriever = None,
            atom_extractor: AtomReviser = None,
            atom_reviser: AtomReviser = None,
            model: str = "llama-3.1-70b-instruct",
            binary_output: bool = True,
            add_topic: bool = False
    ):
        """
        Construct the FactVerify instance.

        Args:
            context_retriever: ContextRetriever
                The service used for retrieving external contexts.
            atom_extractor: AtomExtractor
                The service used for extracting atoms from the response.
            atom_reviser: AtomReviser
                The service used for decontextualizing the atoms.
            model: str
                The name of the model used by FactVerify.
            binary_output: bool
                If true, the output labels are [S - Supported, NS - NotSupported].
                Otherwise, the output labels are [S - Supported, C - Contradicted, U - Inconclusive or Undediced]
            add_topic: bool
                If true, then the topic is added.
        """

        self.query = None
        self.response = None
        self.topic = None
        
        self.model = model

        self.context_retriever = context_retriever
        self.atom_extractor = atom_extractor
        self.atom_reviser = atom_reviser

        self.binary_output = binary_output # default is True
        self.add_topic = add_topic # default is False (relevant only for Biographies)

        # Set up the RITS model
        self.rits_model_info = RITS_MODELS[model]
        self.prompt_template = self.rits_model_info.get("prompt_template", None)
        self.max_new_tokens = self.rits_model_info.get("max_new_tokens", None)
        self.api_base = self.rits_model_info.get("api_base", None)
        self.model_id = self.rits_model_info.get("model_id", None)
        self.prompt_begin = self.rits_model_info.get("prompt_begin", DEFAULT_PROMPT_BEGIN)
        self.prompt_end = self.rits_model_info.get("prompt_end", DEFAULT_PROMPT_END)
        self.use_short_prompt = True if self.max_new_tokens <= 4096 else False
        assert self.prompt_template is not None \
            and self.max_new_tokens is not None \
            and self.api_base is not None \
            and self.model_id is not None

        if not os.environ.get("_DOTENV_LOADED"):
            load_dotenv(override=True) 
            os.environ["_DOTENV_LOADED"] = "1"
        
        self.RITS_API_KEY = os.getenv("RITS_API_KEY")
        print(f"[FactVerify] Using LLM on RITS: {self.model_id}")
        print(f"[FactVerify] Using short prompt: {self.use_short_prompt}")
        print(f"[FactVerify] Binary output: {self.binary_output}")

        self.atoms = {} # indexed by atom id
        self.contexts = {} # indexed by context id

        self.labels_human = None

    def from_dict_with_contexts(
            self,
            data: dict,
    ):
        """
        Create a problem instance from a dict containing both atoms and contexts.

        Args:
            data: str
                The path to the json file containing the problem instance.
        """

        self.query = data["input"]
        self.response = data["output"]
        if self.add_topic:
            self.topic = data["topic"]
        
        print(f"[FactVerify] Reading the human annotated atoms ...")                
        gold_labels = []
        atom_ids = []
        self.atoms = {}
        self.contexts = {}
        atom2contexts = {}
        for elem in data["atoms"]:
            aid = elem["id"]
            text = elem["text"]
            original = elem["original"]
            label = elem["label"]
            contexts = elem["contexts"]
            a = Atom(id=aid, text=text, label=label)
            a.set_original(original)
            atom_ids.append(aid)
            gold_labels.append(label)
            self.atoms[aid] = a
            atom2contexts[aid] = contexts
        print(f"[FactVerify] Labeled atoms found (S, NS): {len(self.atoms)}")
        for _, atom in self.atoms.items():
            print(atom)
        self.labels_human = dict(zip(atom_ids, gold_labels))

        print(f"[FactVerify] Reading the contexts ...")
        for elem_dict in data["contexts"]:
            cid = elem_dict["id"]
            title = elem_dict["title"]
            text = elem_dict["text"]
            snippet = elem_dict.get("snippet", "")
            link = elem_dict.get("link", "")
            ctxt = Context(
                id=cid, 
                atom=None, 
                text=text, 
                title=title, 
                snippet=snippet, 
                link=link
            )
            self.contexts[cid] = ctxt

        print(f"[FactVerify] Contexts loaded: {len(self.contexts)}")
        for aid, atom in self.atoms.items():
            ctxts = []
            for c in atom2contexts[aid]:
                ctxts.append(self.contexts[c])
                self.contexts[c].set_atom(atom)
            atom.add_contexts(ctxts)
        return True

    def build(
            self,
            top_k: int = 5,
            debug_mode: bool = False,
            has_atoms: bool = False,
            has_contexts: bool = False,
            decontextualize_atoms: bool = True,
            no_contexts: bool = False
    ):
        """
        Build the atoms and contexts using the retrieval service.

        Args:
            top_k: int
                Top k most relevant retrieved contexts.
            debug_mode: bool
                Boolean flag indicating debugging mode (default False)
            has_atoms: bool
                A boolean flag indicating if the atoms have already been created.
            has_contexts: bool
                A boolean flag indicating if the contexts have already been created.
            decontextualize_atoms: bool
                A boolean flag indicating that the atoms need to be decontextualized
                (i.e., pronouns he, she, it, ... replaced by the actual entity)
            no_contexts: bool
                A boolean flag indicating if contexts are to be retrieved or not.
                If True, then we run a version that only leverages the internal
                knowledge of the language model.
        """

        # Initialize the scorer
        self.top_k = top_k
        self.debug_mode = debug_mode
        self.no_contexts = no_contexts

        # Create the atomizer (for the response)
        assert self.atom_extractor is not None, f"Atom extractor must be created."
        assert self.atom_reviser is not None, f"Atom reviser must be created."

        print(f"[FactVerify] Building the pipeline ...]")
        print(f"[FactVerify] Using contexts: {not no_contexts}")
        
        # Build the atoms 
        if has_atoms == False:
            self.atoms = build_atoms(
                response=self.response,
                atom_extractor=self.atom_extractor
            )

        assert len(self.atoms) > 0, f"Atoms must be initialized if `has_atoms` is True!"

        # Decontextualize the atoms
        if decontextualize_atoms:
            print(f"[FactVerify] Decontextualize the atoms ...")
            atom_ids = [aid for aid in sorted(self.atoms.keys())]
            old_atoms = [self.atoms[aid].get_text() for aid in atom_ids]
            result = self.atom_reviser.run(old_atoms, self.response)
            for i, aid in enumerate(atom_ids):
                elem = result[i]
                self.atoms[aid].set_text(elem["revised_atom"])
                print(self.atoms[aid])


        # Build the contexts (per atom)
        if no_contexts:
            self.contexts = {}
        else:
            if has_contexts == False: # check if contexts already in file
                self.contexts = build_contexts(
                    atoms=self.atoms,
                    retriever=self.context_retriever,
                )

    def make_prompt(
            self,
            atom: str,
            search_results: List[dict],
            example_data: List[dict]
    ):
        """
        Create the prompt for predicting the label of the atom given the
        search results and the examples.

        Args:
            atom: str
                The string representing the atom.
            search_results: List[dict]
                A list of dictionaries representing the search results 
                relevant to the atom. Each result is a dict with 3 elements:
                title - title of the article, snippet - short text about the article
                 and link - link to the article.
            few_shots: List[dict]
                The examples needed to build the few-shots prompt.

        Returns:
            A string representing the prompt (similar to the one from the VeriScore paper).
        """

        # Set the few-shots section
        element_lst = []
        for dict_item in example_data:
            claim = dict_item["claim"]
            search_result_str = dict_item["search_result"]
            human_label = dict_item["human_label"]
            element_lst.extend([claim, search_result_str, human_label])

        i = 1
        search_results_str = ""
        for dict_item in search_results:
            title = dict_item["title"]
            snippet = dict_item["snippet"]
            link = dict_item["link"]
            search_results_str += f"Search Result {i}\nTitle: {title}\nContent: {snippet}\nLink: {link}\n\n"
            i += 1

        # Set the search results and atom sections
        if self.use_short_prompt: # check for small context (e.g., granite-3.0)
            prompt = VERISCORE_PROMPT_SHORT.format(
                _PROMPT_BEGIN_PLACEHOLDER=self.prompt_begin,
                _PROMPT_END_PLACEHOLDER=self.prompt_end,
                _CLAIM_PLACEHOLDER=atom,
                _SEARCH_RESULTS_PLACEHOLDER=search_results_str,
                *element_lst
            )
        else:
            prompt = VERISCORE_PROMPT.format(
                _PROMPT_BEGIN_PLACEHOLDER=self.prompt_begin,
                _PROMPT_END_PLACEHOLDER=self.prompt_end,
                _CLAIM_PLACEHOLDER=atom,
                _SEARCH_RESULTS_PLACEHOLDER=search_results_str,
                *element_lst
            )
        
        return prompt

    def extract_label(self, text: str) -> str:
        """
        Extract the atom label from the generated text. We expect the label to
        be on the last line of the response, and be one of the following:
            [Supported], [Contradicted], [Inconclusive].
        
        Depending on the binary_output flag we return:
            - [Supported]/S atoms, the others will be [NotSupported]/NS.
            - [Supported]/S atoms, [Contradicted]/C atoms and [Inconclusive]/U atoms.
        """

        label = extract_last_wrapped_response(text)
        if self.binary_output:
            if len(label) > 0 and label.lower() in ['supported']:
                return "S"
            else:
                return "NS"
        else:
            if len(label) > 0 and label.lower() in ['supported']:
                return "S"
            elif len(label) > 0 and label.lower() in ['contradicted']:
                return "C"
            else:
                return "U"
       
    def predict_atom_labels(self) -> dict:
        """
        Use a strong LLM to predict the label S or NS of an atom given its contexts.
        """

        assert len(self.atoms) > 0

        # Use the LLM to label the atom
        print(f"[FactVerify] Labeling atoms with {self.model_id} ...")
        prompts = []
        atom_ids = []

        # Create the prompts for each of the atoms
        for aid, atom in self.atoms.items():
            atom_ids.append(aid)
            contexts = atom.get_contexts()
            if contexts is not None and len(contexts) > 0:
                search_results = [dict(title=c.get_title(), snippet=c.get_snippet(), link=c.get_link()) for c in contexts]
            else:
                search_results = [] # no search results retrieved for the atom

            if self.use_short_prompt:
                prompt = self.make_prompt(
                    atom=atom.get_text(),
                    search_results=search_results,
                    example_data=FEW_SHOTS_SHORT
                )
            else:
                prompt = self.make_prompt(
                    atom=atom.get_text(),
                    search_results=search_results,
                    example_data=FEW_SHOTS
                )

            prompts.append(prompt)

        print(f"[FactVerify] Prompts created: {len(prompts)}")

        # Prepare the LLM call
        results = []
        messages = [[dict(role="user", content=prompt)] for prompt in prompts]
        for _, response in tqdm(
            enumerate(
                litellm.batch_completion(
                    model=self.model_id,
                    api_base=self.api_base,
                    messages=messages,
                    temperature=0,
                    seed=42,
                    api_key=self.RITS_API_KEY,
                    extra_headers={
                        "RITS_API_KEY": self.RITS_API_KEY
                    }
                )
            ),
            total=len(messages),
            desc="Prediction",
            unit="prompts",
            ):
                results.append(response.choices[0].message.content)

        # Postprocess the generated answers
        atom_labels = [self.extract_label(text) for text in results]
        return dict(zip(atom_ids, atom_labels))
    
    def score(self):
        """
        Compute the factuality score taking into consideration the contexts 
        retrieved for each of the atom in the answer.

        Factuality score = # atoms(true) / # atoms

        Intuitively, a score of 100% means that all atoms in the answer are
        factually correct. If none of them are correct, then the score is 0%. If
        only half of the atoms are correct, then the score is 50%.

        Returns:
            dict
                The results dictionary containing the factuality score i.e., a real value in [0, 1]
        """

        # Compute the factuality score (i.e., precision)
        num_true_atoms = 0
        num_false_atoms = 0
        num_uniform_atoms = 0
        labels = self.predict_atom_labels()
        for _, label in labels.items():
            if self.binary_output:
                if label == "S":
                    num_true_atoms += 1
                else:
                    num_false_atoms += 1
            else:
                if label == "S":
                    num_true_atoms += 1
                elif label == "C":
                    num_false_atoms += 1
                else:
                    num_uniform_atoms += 1
        
        # Precision
        fscore = float(num_true_atoms)/float(len(self.atoms))

        results = {}
        results["factuality_score"] = fscore
        results["num_atoms"] = len(self.atoms)
        results["num_contexts"] = len(self.contexts)
        results["num_true_atoms"] = num_true_atoms
        results["num_false_atoms"] = num_false_atoms
        results["num_uniform_atoms"] = num_uniform_atoms
        results["entropy"] = None
        results["avg_entropy"] = None

        print(f"[FactVerify] Predictions: {labels}")
        if self.labels_human is not None and self.binary_output is True:
            true_atoms = 0
            false_atoms = 0
            num_true_positive = 0
            num_true_negative = 0
            num_false_positive = 0
            num_false_negative = 0
            for aid, l in self.labels_human.items():
                if l == "S":
                    true_atoms += 1
                    if labels[aid] == "S":
                        num_true_positive += 1
                    else:
                        num_false_negative += 1
                else:
                    false_atoms += 1
                    if labels[aid] == "NS":
                        num_true_negative += 1
                    else:
                        num_false_positive += 1     

            fscore_gold = true_atoms/len(self.labels_human)
            print(f"[FactVerify] Gold labels: {self.labels_human}")
            print(f"[FactVerify] Predictions: {labels}")
            print(f"[FactVerify] Gold fscore: {fscore_gold} ({true_atoms}/{len(self.labels_human)})")
            results["gold_factuality_score"] = fscore_gold
            results["gold_true_atoms"] = true_atoms
            results["true_positive"] = num_true_positive
            results["true_negative"] = num_true_negative
            results["false_positive"] = num_false_positive
            results["false_negative"] = num_false_negative
        elif self.labels_human is not None and self.binary_output is False:
            true_atoms = 0
            false_atoms = 0
            num_true_positive = 0
            num_true_negative = 0
            num_false_positive = 0
            num_false_negative = 0
            for aid, l in self.labels_human.items(): # true labels are either S or NS
                if l == "S":
                    true_atoms += 1
                    if labels[aid] == "S":
                        num_true_positive += 1
                    else:
                        num_false_negative += 1
                else:
                    false_atoms += 1
                    if labels[aid] in ["C", "U"]:
                        num_true_negative += 1
                    else:
                        num_false_positive += 1     

            fscore_gold = true_atoms/len(self.labels_human)
            print(f"[FactVerify] Gold labels: {self.labels_human}")
            print(f"[FactVerify] Predictions: {labels}")
            print(f"[FactVerify] Gold fscore: {fscore_gold} ({true_atoms}/{len(self.labels_human)})")
            results["gold_factuality_score"] = fscore_gold
            results["gold_true_atoms"] = true_atoms
            results["true_positive"] = num_true_positive
            results["true_negative"] = num_true_negative
            results["false_positive"] = num_false_positive
            results["false_negative"] = num_false_negative


        if self.topic is not None and len(self.topic) > 0:
            results["topic"] = self.topic
        results["input"] = self.query

        return results

def test():

    model = "granite-3.0-8b-instruct"

    context_retriever = ContextRetriever(service_type="google")
    atom_extractor = AtomExtractor(model)
    atom_reviser = AtomReviser(model)

    # Create the FactVerify pipeline
    pipeline = FactVerify(
        context_retriever=context_retriever,
        atom_extractor=atom_extractor,
        atom_reviser=atom_reviser,
        model=model
    )

    # Load the problem instance from a file
    json_file = "/home/radu/git/fm-factual/examples/test2.json"
    with open(json_file, "r") as f:
        data = json.load(f)

    pipeline.from_dict_with_contexts(data)

    # Build the FactVerify pipeline
    pipeline.build(
        top_k=5,
        has_atoms=True,
        has_contexts=True,
        decontextualize_atoms=False
    )

    results = pipeline.score()
    print(f"[FactVerify] Results: {results}")
    print(f"Done.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_file', 
        type=str, 
        default=None, 
        help="Path to the file containing the input dataset."
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default=None, 
        help="Path to the output directory."
    )

    parser.add_argument(
        '--dataset_name', 
        type=str, 
        default=None, 
        help="Name of the dataset."
    )

    parser.add_argument(
        '--model', 
        type=str, 
        default="llama-3.1-70b-instruct", 
        help="Name of the underlying LLM."
    )

    parser.add_argument(
        '--service_type', 
        type=str,
        default="google", 
        help="Retriever type (langchain or google only in this case)."
    )

    parser.add_argument(
        '--binary_output', 
        default=False, 
        action='store_true', 
        help="Ensure binary output for the atomic unit label prediction."
    )

    parser.add_argument(
        '--add_topic', 
        default=False, 
        action='store_true', 
        help="Ensure the the topic is added (relevant only for Biographies)."
    )

    parser.add_argument(
        '--test', 
        default=False, 
        action='store_true', 
        help="Debugging mode."
    )

    parser.add_argument(
        '--no_contexts', 
        default=False, 
        action='store_true', 
        help="Flag for enabling FactScore Zero, without contexts."
    )

    args = parser.parse_args()

    if args.test == True:
        test()
        sys.exit(0)

    option = "0" if args.no_contexts else ""

    # Create the atom extractor, atom reviser and context retriever
    context_retriever = ContextRetriever(service_type="google", db_remote=False)
    atom_extractor = AtomExtractor(model=args.model)
    atom_reviser = AtomReviser(model=args.model)

    print(f"[FactVerify] Processing dataset: {args.input_file}")
    filename = args.input_file # a jsonl file

    with open(filename) as f:
        lines = f.read().splitlines()
    df_inter = pd.DataFrame(lines)
    df_inter.columns = ['json_element']
    df_inter['json_element'].apply(json.loads)
    df = pd.json_normalize(df_inter['json_element'].apply(json.loads))
    dataset = df.to_dict('records')
    f.close()

    print(f"[FactVerify] Loading data from: {filename}")
    print(f"[FactVerify] Found {len(dataset)} elements")

    # Check if previous results exist. If yes, load them and skip over them
    # when processing the input dataset.
    filename = "eval_results_factverify{}_{}_{}_{}.jsonl".format(
        option,
        args.service_type,
        args.dataset_name,
        args.model
    )
    output_filename = os.path.join(args.output_dir, filename)
    print(f"[FactVerify] Reading previous results from: {output_filename}")
    evaluation_data = []
    if os.path.isfile(output_filename):
        with open(output_filename, "r") as f:
            lines = f.readlines()
            for line in lines:
                evaluation_data.append(json.loads(line))

    # Evaluate the dataset
    print(f"[FactVerify] Found {len(evaluation_data)} existing evaluations data.")
    for data in dataset:

        # Check if current data has been processed already
        processed = False
        for eval_data in evaluation_data:
            if eval_data["input"] == data["input"]:
                processed = True
                break
        if processed:
            print(f"[FactVerify] Input {data} already processed.")
            continue

        # Process the data point with FactVerify pipeline
        pipeline = FactVerify(
            context_retriever=context_retriever,
            atom_extractor=atom_extractor,
            atom_reviser=atom_reviser,
            model=args.model,
            add_topic=args.add_topic,
            binary_output=args.binary_output
        )

        # Load the problem instance from a file
        ok = pipeline.from_dict_with_contexts(data)
        if not ok:
            continue # annotations are null (ignore)

        # Build the FactVerify pipeline
        pipeline.build(
            top_k=5,
            has_atoms=True,
            has_contexts=True,
            decontextualize_atoms=False,
            no_contexts=args.no_contexts
        )

        results = pipeline.score()
        results["model_name"] = args.model
        evaluation_data.append(results)
        print(f"[FactVerify] Results: {results}")

        # Save results to a file (progressively)
        filename = "eval_results_factverify{}_{}_{}_{}.jsonl".format(
            option,
            args.service_type,
            args.dataset_name,
            args.model
        )
        output_filename = os.path.join(args.output_dir, filename)
        print(f"[FactVerify] Writing results to: {output_filename}")
        with open(output_filename, "w") as f:
            for res in evaluation_data:
                f.write(f"{json.dumps(res)}\n")
        f.close()

    print("Done.")
