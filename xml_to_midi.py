import os
from music21 import converter

def convert_xml_to_midi_in_folders(input_folder, output_folder):
    """
    지정된 입력 폴더의 모든 XML 파일을 MIDI로 변환하여
    지정된 출력 폴더에 저장합니다.

    Args:
        input_folder (str): XML 파일들이 있는 폴더 경로
        output_folder (str): MIDI 파일을 저장할 폴더 경로
    """
    
    if not os.path.isdir(input_folder):
        print(f"오류: '{input_folder}' 입력 폴더를 찾을 수 없습니다.")
        return

    # 출력 폴더가 없으면 새로 생성
    if not os.path.exists(output_folder):
        print(f"'{output_folder}' 출력 폴더가 존재하지 않아 새로 생성합니다.")
        os.makedirs(output_folder)

    # 입력 폴더 내의 모든 파일 목록을 가져옴
    for filename in os.listdir(input_folder):
        # 파일 확장자가 '.xml'인지 확인
        if filename.endswith('.xml'):
            
            # 전체 파일 경로 생성 (입력)
            xml_path = os.path.join(input_folder, filename)
            
            # MIDI 파일명 및 경로 생성 (출력)
            midi_filename = filename.replace('.xml', '.mid')
            midi_path = os.path.join(output_folder, midi_filename)
            
            print(f"'{filename}' 파일을 변환 중...")

            try:
                # MusicXML 파일 로드
                score = converter.parse(xml_path)
                
                # MIDI 파일로 저장
                score.write('midi', fp=midi_path)
                print(f"-> '{midi_filename}' 파일로 저장 완료.")

            except Exception as e:
                print(f"알 수 없는 오류: '{filename}' 파일 처리 중 오류 발생 - {e}")

# 변환할 XML 파일들이 있는 폴더 경로
input_folder = '/Users/simhyeongju/AVAPT/data/Mikrokosmos-difficulty/musicxml'

# MIDI 파일이 저장될 폴더 경로
output_folder = '/Users/simhyeongju/AVAPT/data/Mikrokosmos-difficulty/midi'

# 함수 실행
convert_xml_to_midi_in_folders(input_folder, output_folder)

print("\n모든 작업이 완료되었습니다!")