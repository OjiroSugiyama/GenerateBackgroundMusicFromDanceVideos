# Motion-Aware Sequencer

Motion-Aware Sequencer is a loop sequencer based on the excitement of the video.

## Requirements

* OS : Linux or Windows
* Python 3.6
* VLC media player  
<https://www.videolan.org/index.html>
* FFmpeg  
<https://www.ffmpeg.org/>

## Installation

1. Clone the repository.
1. If necessary, install python version management system and switch to python 3.6.
1. If necessary, create and activate a new virtual environment.  
**Create virtual environment**  
`python -m venv [venv dir name]`  
**Activate virtual environment**  
Linux `source [venv dir name]/bin/activate`  
Windows `[venv dir name]\Scripts\activate`  

1. Install dependencies with requirements.txt.
1. Purchase the following sound sources.  
  Sound PooL vol.1  
  <https://www.ah-soft.com/soundpool/#rewire01>
1. Copy the wav files to the target directory.

|Directory|File|
|---|---|
|TechnoTrance/Bass/0|EightBass 1.wav|
||EightBass 2.wav|
||EightBass 3.wav|
||EightBass 4.wav|
||EightBass 5.wav|
||EightBass 6.wav|
||EightBass 7.wav|
||SoftDeep 1.wav|
||SoftDeep 2.wav|
||SoftDeep 3.wav|
||SoftDeep 4.wav|
||SoftDeep 5.wav|
||SoftDeep 6.wav|
||SoftDeep 7.wav|
|TechnoTrance/Bass/1|Hardclick 1.wav|
||Hardclick 2.wav|
||Hardclick 3.wav|
||Hardclick 4.wav|
||Hardclick 5.wav|
||Hardclick 6.wav|
||Hardclick 7.wav|
||TB Bass 1.wav|
||TB Bass 2.wav|
||TB Bass 3.wav|
||TB Bass 4.wav|
||TB Bass 5.wav|
||TB Bass 6.wav|
||TB Bass 7.wav|
|TechnoTrance/Bass/2|MaasedBass 1.wav|
||MaasedBass 2.wav|
||MaasedBass 3.wav|
||MaasedBass 4.wav|
||MaasedBass 5.wav|
||MaasedBass 6.wav|
||MaasedBass 7.wav|
||OfflineA 1.wav|
||OfflineA 2.wav|
||OfflineA 3.wav|
||OfflineA 4.wav|
||OfflineA 5.wav|
||OfflineA 6.wav|
||OfflineA 7.wav|
||SawBass 1.wav|
||SawBass 2.wav|
||SawBass 3.wav|
||SawBass 4.wav|
||SawBass 5.wav|
||SawBass 6.wav|
||SawBass 7.wav|
||TB Off 1.wav|
||TB Off 2.wav|
||TB Off 3.wav|
||TB Off 4.wav|
||TB Off 5.wav|
||TB Off 6.wav|
||TB Off 7.wav|
|TechnoTrance/Bass/3|DanceOff 1.wav|
||DanceOff 2.wav|
||DanceOff 3.wav|
||DanceOff 4.wav|
||DanceOff 5.wav|
||DanceOff 6.wav|
||DanceOff 7.wav|
||OfflineB 1.wav|
||OfflineB 2.wav|
||OfflineB 3.wav|
||OfflineB 4.wav|
||OfflineB 5.wav|
||OfflineB 6.wav|
||OfflineB 7.wav|
||Simplic 1.wav|
||Simplic 2.wav|
||Simplic 3.wav|
||Simplic 4.wav|
||Simplic 5.wav|
||Simplic 6.wav|
||Simplic 7.wav|
||Solar 1.wav|
||Solar 2.wav|
||Solar 3.wav|
||Solar 4.wav|
||Solar 5.wav|
||Solar 6.wav|
||Solar 7.wav|
|TechnoTrance/Bass/4|RoomBass 1.wav|
||RoomBass 2.wav|
||RoomBass 3.wav|
||RoomBass 4.wav|
||RoomBass 5.wav|
||RoomBass 6.wav|
||RoomBass 7.wav|
||SeqBass 1.wav|
||SeqBass 2.wav|
||SeqBass 3.wav|
||SeqBass 4.wav|
||SeqBass 5.wav|
||SeqBass 6.wav|
||SeqBass 7.wav|
||ShuffleBass 1.wav|
||ShuffleBass 2.wav|
||ShuffleBass 3.wav|
||ShuffleBass 4.wav|
||ShuffleBass 5.wav|
||ShuffleBass 6.wav|
||ShuffleBass 7.wav|
|TechnoTrance/Drums/0|Hard Beat F.wav|
||Hard Beat G.wav|
||Hard Beat H.wav|
||Knarzer Beat Q.wav|
||Knarzer Beat R.wav|
||Knarzer Beat S.wav|
||Knarzer Beat T.wav|
||Knarzer Beat U.wav|
||Knarzer Beat V.wav|
||Knarzer Beat W.wav|
||Knarzer Beat X.wav|
||Knarzer Beat Y.wav|
||Mini Beat F.wav|
||Mini Beat G.wav|
||Shuffle Beat L.wav|
||Shuffle Beat M.wav|
||Tchacka Beat Q.wav|
||Tchacka Beat R.wav|
||Tchacka Beat T.wav|
||Tchacka Beat U.wav|
|TechnoTrance/Drums/1|Hard Beat E.wav|
||Knarzer Beat G.wav|
||Knarzer Beat I.wav|
||Knarzer Beat O.wav|
||Knarzer Beat P.wav|
||Knarzer Beat Z.wav|
||Shuffle Beat I.wav|
||Shuffle Beat J.wav|
||Tchacka Beat S.wav|
|TechnoTrance/Drums/2|Hard Beat D.wav|
||Knarzer Beat E.wav|
||Knarzer Beat M.wav|
||Knarzer Beat N.wav|
|TechnoTrance/Drums/3|Hard Beat B.wav|
||Hard Beat C.wav|
||Knarzer Beat C.wav|
||Knarzer Beat K.wav|
||Knarzer Beat L.wav|
|TechnoTrance/Drums/4|Hard Beat A.wav|
||Knarzer Beat B.wav|
||Knarzer Beat D.wav|
||Knarzer Beat F.wav|
||Knarzer Beat H.wav|
||Knarzer Beat J.wav|
|TechnoTrance/Sequence/0|Bleep 1.wav|
||Bleep 2.wav|
||Bleep 3.wav|
||Bleep 4.wav|
||Bleep 5.wav|
||Bleep 6.wav|
||Bleep 7.wav|
||DistTB B 1.wav|
||DistTB B 2.wav|
||DistTB B 3.wav|
||DistTB B 4.wav|
||DistTB B 5.wav|
||DistTB B 6.wav|
||DistTB B 7.wav|
||SequenceA 1.wav|
||SequenceA 2.wav|
||SequenceA 3.wav|
||SequenceA 4.wav|
||SequenceA 5.wav|
||SequenceA 6.wav|
||SequenceA 7.wav|
||Smashed 1.wav|
||Smashed 2.wav|
||Smashed 3.wav|
||Smashed 4.wav|
||Smashed 5.wav|
||Smashed 6.wav|
||Smashed 7.wav|
||TB Wow 1.wav|
||TB Wow 2.wav|
||TB Wow 3.wav|
||TB Wow 4.wav|
||TB Wow 5.wav|
||TB Wow 6.wav|
||TB Wow 7.wav|
||Wow 1.wav|
||Wow 2.wav|
||Wow 3.wav|
||Wow 4.wav|
||Wow 5.wav|
||Wow 6.wav|
||Wow 7.wav|
|TechnoTrance/Sequence/1|Belzebub 1.wav|
||Belzebub 2.wav|
||Belzebub 3.wav|
||Belzebub 4.wav|
||Belzebub 5.wav|
||Belzebub 6.wav|
||Belzebub 7.wav|
||DistTB 1.wav|
||DistTB 2.wav|
||DistTB 3.wav|
||DistTB 4.wav|
||DistTB 5.wav|
||DistTB 6.wav|
||DistTB 7.wav|
||Hard Edge 1.wav|
||Hard Edge 2.wav|
||Hard Edge 3.wav|
||Hard Edge 4.wav|
||Hard Edge 5.wav|
||Hard Edge 6.wav|
||Hard Edge 7.wav|
||TB Add 1.wav|
||TB Add 2.wav|
||TB Add 3.wav|
||TB Add 4.wav|
||TB Add 5.wav|
||TB Add 6.wav|
||TB Add 7.wav|
||TB Answer 1.wav|
||TB Answer 2.wav|
||TB Answer 3.wav|
||TB Answer 4.wav|
||TB Answer 5.wav|
||TB Answer 6.wav|
||TB Answer 7.wav|
|TechnoTrance/Sequence/2|GoSeq 1.wav|
||GoSeq 2.wav|
||GoSeq 3.wav|
||GoSeq 4.wav|
||GoSeq 5.wav|
||GoSeq 6.wav|
||GoSeq 7.wav|
||TB 3003 B 1.wav|
||TB 3003 B 2.wav|
||TB 3003 B 3.wav|
||TB 3003 B 4.wav|
||TB 3003 B 5.wav|
||TB 3003 B 6.wav|
||TB 3003 B 7.wav|
||TB 3003 C 1.wav|
||TB 3003 C 2.wav|
||TB 3003 C 3.wav|
||TB 3003 C 4.wav|
||TB 3003 C 5.wav|
||TB 3003 C 6.wav|
||TB 3003 C 7.wav|
|TechnoTrance/Sequence/3|Backing 1.wav|
||Backing 2.wav|
||Backing 3.wav|
||Backing 4.wav|
||Backing 5.wav|
||Backing 6.wav|
||Backing 7.wav|
||Hackerseq 1.wav|
||Hackerseq 2.wav|
||Hackerseq 3.wav|
||Hackerseq 4.wav|
||Hackerseq 5.wav|
||Hackerseq 6.wav|
||Hackerseq 7.wav|
||Romancer 1.wav|
||Romancer 2.wav|
||Romancer 3.wav|
||Romancer 4.wav|
||Romancer 5.wav|
||Romancer 6.wav|
||Romancer 7.wav|
||TB 3003 1.wav|
||TB 3003 2.wav|
||TB 3003 3.wav|
||TB 3003 4.wav|
||TB 3003 5.wav|
||TB 3003 6.wav|
||TB 3003 7.wav|
|TechnoTrance/Sequence/4|Faster 1.wav|
||Faster 2.wav|
||Faster 3.wav|
||Faster 4.wav|
||Faster 5.wav|
||Faster 6.wav|
||Faster 7.wav|
||TB Wild 1.wav|
||TB Wild 2.wav|
||TB Wild 3.wav|
||TB Wild 4.wav|
||TB Wild 5.wav|
||TB Wild 6.wav|
||TB Wild 7.wav|
|TechnoTrance/Synth/0|TB Dist 1.wav|
||TB Dist 2.wav|
||TB Dist 3.wav|
||TB Dist 4.wav|
||TB Dist 5.wav|
||TB Dist 6.wav|
||TB Dist 7.wav|
|TechnoTrance/Synth/1|Stack A 1.wav|
||Stack A 2.wav|
||Stack A 3.wav|
||Stack A 4.wav|
||Stack A 5.wav|
||Stack A 6.wav|
||Stack A 7.wav|
|TechnoTrance/Synth/2|Shrabnelle 1.wav|
||Shrabnelle 2.wav|
||Shrabnelle 3.wav|
||Shrabnelle 4.wav|
||Shrabnelle 5.wav|
||Shrabnelle 6.wav|
||Shrabnelle 7.wav|
|TechnoTrance/Synth/3|AlarmLead 1.wav|
||AlarmLead 2.wav|
||AlarmLead 3.wav|
||AlarmLead 4.wav|
||AlarmLead 5.wav|
||AlarmLead 6.wav|
||AlarmLead 7.wav|
||BigLead Low 1.wav|
||BigLead Low 2.wav|
||BigLead Low 3.wav|
||BigLead Low 4.wav|
||BigLead Low 5.wav|
||BigLead Low 6.wav|
||BigLead Low 7.wav|
|TechnoTrance/Synth/4|BigLead High 1.wav|
||BigLead High 2.wav|
||BigLead High 3.wav|
||BigLead High 4.wav|
||BigLead High 5.wav|
||BigLead High 6.wav|
||BigLead High 7.wav|
||Lead 1.wav|
||Lead 2.wav|
||Lead 3.wav|
||Lead 4.wav|
||Lead 5.wav|
||Lead 6.wav|
||Lead 7.wav|
||Leader 1.wav|
||Leader 2.wav|
||Leader 3.wav|
||Leader 4.wav|
||Leader 5.wav|
||Leader 6.wav|
||Leader 7.wav|

## Usage

### Activate pyenv

* `source pyenv/bin/activate`

### Run the program

* `python main.py`
* Click "open" and select a mp4 file from the Movie directory.
* Click "read" to analyze the movie. When the analysis process is over, click "refrect".
* Click "play" to play the movie and composed music.

#### Buttons

**create**  Create music  
**play**  Play selected movie & composed music  
**pause**  Pause movie and music  
**stop**  Stop movie and music  

**no fix / fix**  Align/Do not align sound material  
**nopart / auto**  Consider/Do not consider the composition of the music  

**open**  Open the movie folder and select a movie
**read**      Analyze selected movie
**cancel**   Suspend movie analysis
**reflect**   Reflect the analysis result of the movie on the curve

### Deactivate pyenv

* `deactivate`

### Demo

<http://www2.kthrlab.jp/members/yasusaka/index/index.html>

## Authors

* Tetsuro Kitahara（Nihon University, Japan）
  * kitahara@kthrlab.jp
  * <https://www.kthrlab.jp/>
* Bunta Yasusaka
  * <http://www2.kthrlab.jp/members/yasusaka/index/index.html>
