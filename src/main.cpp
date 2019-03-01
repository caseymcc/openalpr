/*
* Copyright (c) 2015 OpenALPR Technology, Inc.
* Open source Automated License Plate Recognition [http://www.openalpr.com]
*
* This file is part of OpenALPR.
*
* OpenALPR is free software: you can redistribute it and/or modify
* it under the terms of the GNU Affero General Public License
* version 3 as published by the Free Software Foundation
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU Affero General Public License for more details.
*
* You should have received a copy of the GNU Affero General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#define FIND_MEMORY_LEAK

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#ifdef FIND_MEMORY_LEAK
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif

#include <cstdio>
#include <sstream>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <thread>

#include "tclap/CmdLine.h"
#include "support/filesystem.h"
#include "support/timing.h"
#include "support/platform.h"
#include "video/videobuffer.h"
#include "motiondetector.h"
#include "alpr.h"

#include "cjson.h"

using namespace alpr;

const std::string MAIN_WINDOW_NAME="ALPR main window";

const bool SAVE_LAST_VIDEO_STILL=false;
const std::string LAST_VIDEO_STILL_LOCATION="/tmp/laststill.jpg";
const std::string WEBCAM_PREFIX="/dev/video";
MotionDetector motiondetector;

/** Function Headers */
bool processImageDirectory(Alpr &alpr, cv::Mat &frame, std::string &directory, bool outputJson, bool outputJsonFormated);
bool detectandshow(Alpr* alpr, cv::Mat frame, std::string region, bool writeJson, bool formated=false, std::string jsonFile="");
bool is_supported_image(std::string image_file);

struct Settings
{
    Settings()
    {
        do_motiondetection=true;

        configFile="";
        outputJson=false;
        outputJsonFormated=false;
        seektoms=0;
        detectRegion=false;
        debug_mode=false;
        numberOfThreads=1;

        measureProcessingTime=false;

        program_active=true;
        processedImages=0;
    }

    bool do_motiondetection;

    std::string configFile;
    bool outputJson;
    bool outputJsonFormated;
	bool skipProcessed;
    int seektoms;
    bool detectRegion;
    std::string country;
    int topn;
    bool debug_mode;
    int numberOfThreads;

    bool measureProcessingTime;
    std::string templatePattern;

    bool program_active;
    int processedImages;
};

struct ProcessRequest
{
    std::string fileName;
    cv::Mat frame;
    AlprResults results;
    double processingTime;
};

struct ThreadInfo
{
    ThreadInfo() {}

    bool running;
    std::condition_variable processEvent;
    std::condition_variable completeEvent;

    //the following should always be guarded by the lock
    std::mutex processMutex;
    std::deque<ProcessRequest *> queued;
    std::deque<ProcessRequest *> completed;
};

enum MediaType
{
    unknown,
    init,
    buffer,
    stdin_,
    webcam,
    url,
    video,
    image,
    directory
};

struct CustomState
{};

struct State
{
    size_t fileNameIndex;
    std::vector<std::string> fileNames;
    std::vector<std::shared_ptr<CustomState>> stack;

    MediaType type;
};
//thread functions
void processThread(Settings &settings, ThreadInfo *threadInfo);
bool getRequest(const Settings &settings, State &state, ProcessRequest *request);

//log functions
void writeJson(std::string jsonFile, AlprResults &results);
void toConsole(Settings &settings, ProcessRequest *request);
std::string toJson(const AlprResults results, bool formated);

int main(int argc, const char** argv)
{
    std::vector<std::thread> threads;

#ifdef FIND_MEMORY_LEAK
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF|_CRTDBG_LEAK_CHECK_DF);

    //	_crtBreakAlloc = 165;
    _CrtSetBreakAlloc(165);
#endif
    std::vector<std::string> filenames;
    Settings settings;
    State state;
    ThreadInfo threadInfo;

    TCLAP::CmdLine cmd("OpenAlpr Command Line Utility", ' ', Alpr::getVersion());
    TCLAP::UnlabeledMultiArg<std::string>  fileArg("image_file", "Image containing license plates", true, "", "image_file_path");
    TCLAP::ValueArg<std::string> countryCodeArg("c", "country", "Country code to identify (either us for USA or eu for Europe).  Default=us", false, "us", "country_code");
    TCLAP::ValueArg<int> seekToMsArg("", "seek", "Seek to the specified millisecond in a video file. Default=0", false, 0, "integer_ms");
    TCLAP::ValueArg<std::string> configFileArg("", "config", "Path to the openalpr.conf file", false, "", "config_file");
    TCLAP::ValueArg<std::string> templatePatternArg("p", "pattern", "Attempt to match the plate number against a plate pattern (e.g., md for Maryland, ca for California)", false, "", "pattern code");
    TCLAP::ValueArg<int> topNArg("n", "topn", "Max number of possible plate numbers to return.  Default=10", false, 10, "topN");
    TCLAP::ValueArg<int> numberOfThreads("t", "threads", "Number of threads to use.  Default=1", false, 1, "threads");

    TCLAP::SwitchArg jsonSwitch("j", "json", "Output recognition results in JSON format.  Default=off", cmd, false);
	TCLAP::SwitchArg skipSwitch("s", "skip", "Skip images with json files.  Default=off", cmd, false);
    TCLAP::SwitchArg jsonFormatedSwitch("f", "formated_json", "Output JSON results formated.  Default=off", cmd, false);
    TCLAP::SwitchArg debugSwitch("", "debug", "Enable debug output.  Default=off", cmd, false);
    TCLAP::SwitchArg detectRegionSwitch("d", "detect_region", "Attempt to detect the region of the plate image.  [Experimental]  Default=off", cmd, false);
    TCLAP::SwitchArg clockSwitch("", "clock", "Measure/print the total time to process image and all plates.  Default=off", cmd, false);
    TCLAP::SwitchArg motiondetect("", "motion", "Use motion detection on video file or stream.  Default=off", cmd, false);

    try
    {
        cmd.add(templatePatternArg);
        cmd.add(seekToMsArg);
        cmd.add(topNArg);
        cmd.add(numberOfThreads);
        cmd.add(configFileArg);
        cmd.add(fileArg);
        cmd.add(countryCodeArg);

        if(cmd.parse(argc, argv)==false)
        {
            // Error occurred while parsing.  Exit now.
            return 1;
        }

        filenames=fileArg.getValue();

        settings.country=countryCodeArg.getValue();
        settings.seektoms=seekToMsArg.getValue();
        settings.outputJson=jsonSwitch.getValue();
		settings.skipProcessed = skipSwitch.getValue();
        settings.outputJsonFormated=jsonFormatedSwitch.getValue();
        settings.debug_mode=debugSwitch.getValue();
        settings.configFile=configFileArg.getValue();
        settings.detectRegion=detectRegionSwitch.getValue();
        settings.templatePattern=templatePatternArg.getValue();
        settings.topn=topNArg.getValue();
        settings.numberOfThreads=numberOfThreads.getValue();
        settings.measureProcessingTime=clockSwitch.getValue();
        settings.do_motiondetection=motiondetect.getValue();
    }
    catch(TCLAP::ArgException &e)    // catch any exceptions
    {
        std::cerr<<"error: "<<e.error()<<" for arg "<<e.argId()<<std::endl;
        return 1;
    }

    size_t fileIndex=0;
    std::vector<ProcessRequest> processRequests;
    std::vector<ProcessRequest *> freeRequest;
    int inFlight=0;
    std::deque<ProcessRequest *> completed;
    ProcessRequest *request=nullptr;
    bool hasRequest=false;
    int processRequestSize=settings.numberOfThreads*3;

    //create process requests
    processRequests.resize(processRequestSize);
    for(size_t i=0; i<processRequestSize; ++i)
        freeRequest.push_back(&processRequests[i]);

    //start threads
    for(size_t i=0; i<settings.numberOfThreads; ++i)
    {
//        threads.push_back(std::thread(processThread, settings, threadInfo));
        threads.emplace_back(processThread, settings, &threadInfo);
    }

    state.fileNameIndex=0;
    state.fileNames=filenames;
    state.type=MediaType::init;
    
    while(true)
    {
        if((request==nullptr)&&!freeRequest.empty())
        {
            request=freeRequest.back();
            freeRequest.pop_back();
        }

        if(request!=nullptr)
            hasRequest=getRequest(settings, state, request);

        {//for mutex lock
            std::unique_lock<std::mutex> lock(threadInfo.processMutex);

            if(!threadInfo.completed.empty())
            {
                completed.insert(completed.end(), threadInfo.completed.begin(), threadInfo.completed.end());
                inFlight-=threadInfo.completed.size();
                threadInfo.completed.clear();
            }

            if((request==nullptr)&&freeRequest.empty()&&completed.empty())
            {
                threadInfo.completeEvent.wait(lock);
                continue;
            }

            if(hasRequest)
            {
                threadInfo.queued.push_back(request);
                inFlight++;
                hasRequest=false;
                request=nullptr;

                //tell threads somethins ready
                threadInfo.processEvent.notify_all();
            }
        }

        if(!completed.empty())
        {
            //write out results
            for(ProcessRequest *completedRequest:completed)
            {
                freeRequest.push_back(completedRequest);
                inFlight--;

                if(settings.outputJson)
                {
                    std::string directory=get_directory_from_path(completedRequest->fileName);
                    std::string jsonFile=directory+"/"+filenameWithoutExtension(completedRequest->fileName)+".json";

                    writeJson(jsonFile, completedRequest->results);
                }
                toConsole(settings, completedRequest);
            }
            completed.clear();
        }
    }
}

void writeJson(std::string jsonFile, AlprResults &results)
{
    std::string json=toJson(results, true);
//    std::cout<<json<<std::endl;

    if(!jsonFile.empty())
    {
        FILE *file=fopen(jsonFile.c_str(), "w");

        if(file)
        {
            fwrite(json.c_str(), sizeof(char), json.size(), file);
            fclose(file);
        }
    }
}

void toConsole(Settings &settings, ProcessRequest *request)
{
    AlprResults &results=request->results;

    std::cout<<"Image: "<<request->fileName<<"\n";
    for(int i=0; i < results.plates.size(); i++)
    {
        std::cout<<"plate"<<i<<": "<<results.plates[i].topNPlates.size()<<" results";
        if(settings.measureProcessingTime)
            std::cout<<" -- Processing Time = "<<results.plates[i].processing_time_ms<<"ms.";
        std::cout<<std::endl;
    
        if(results.plates[i].regionConfidence > 0)
            std::cout<<"State ID: "<<results.plates[i].region<<" ("<<results.plates[i].regionConfidence<<"% confidence)"<<std::endl;
    
        for(int k=0; k < results.plates[i].topNPlates.size(); k++)
        {
            // Replace the multiline newline character with a dash
            std::string no_newline=results.plates[i].topNPlates[k].characters;
            std::replace(no_newline.begin(), no_newline.end(), '\n', '-');
    
            std::cout<<"    - "<<no_newline<<"\t confidence: "<<results.plates[i].topNPlates[k].overall_confidence;
            if(settings.templatePattern.size() > 0||results.plates[i].regionConfidence > 0)
                std::cout<<"\t pattern_match: "<<results.plates[i].topNPlates[k].matches_template;
    
            std::cout<<std::endl;
        }
    }
}

struct VideoState:CustomState
{
    VideoState():opened(false) {}

    bool opened;
    cv::VideoCapture cap;
    int framenum;
};

struct UrlState:CustomState
{
    UrlState():connected(false) {}
    ~UrlState() { if(connected) videoBuffer.disconnect(); }

    bool connected;
    VideoBuffer videoBuffer;
    int framenum;
};

struct DirectoryState:CustomState
{
    DirectoryState(std::string &directory) { directoryStack.push_back(directory); }

    std::vector<std::string> directoryStack;
    std::string lastFile;
};



bool isVideo(std::string fileName)
{
    if(hasEndingInsensitive(fileName, ".avi")||hasEndingInsensitive(fileName, ".mp4")||
        hasEndingInsensitive(fileName, ".webm")||
        hasEndingInsensitive(fileName, ".flv")||hasEndingInsensitive(fileName, ".mjpg")||
        hasEndingInsensitive(fileName, ".mjpeg")||
        hasEndingInsensitive(fileName, ".mkv")
        )
        return true;
    return false;
}

MediaType getMediaType(std::string &input)
{
    if(input=="-")
        return MediaType::buffer;
    else if(input=="stdin")
        return MediaType::stdin_;
    else if(input=="webcam"||startsWith(input, WEBCAM_PREFIX))
        return MediaType::webcam;
    else if(startsWith(input, "http://")||startsWith(input, "https://"))
        return MediaType::url;
    else if(isVideo(input))
        return MediaType::video;
    else if(is_supported_image(input))
        return MediaType::image;
    else if(DirectoryExists(input.c_str()))
        return MediaType::directory;
    return MediaType::unknown;
}

bool getBuffer(const Settings &settings, CustomState *state, std::string &input, ProcessRequest *request)
{
    std::vector<uchar> data;
    int c;

    while((c=fgetc(stdin))!=EOF)
    {
        data.push_back((uchar)c);
    }

    request->frame=cv::imdecode(cv::Mat(data), 1);
    if(request->frame.empty())
    {
        std::cerr<<"Image invalid: "<<input<<std::endl;
        return false;
    }
    return true;
}

bool getStdIn(const Settings &settings, CustomState *state, std::string &input, ProcessRequest *request)
{
    std::string fileName;

    if(!std::getline(std::cin, fileName))
        return false;

    if(!fileExists(fileName.c_str()))
    {
        std::cerr<<"Image file not found: "<<fileName<<std::endl;
        return false;
    }

    request->frame=cv::imread(fileName);

    if(request->frame.empty())
    {
        std::cerr<<"Image invalid: "<<fileName<<std::endl;
        return false;
    }

    return true;
}

bool getWebCam(const Settings &settings, VideoState *state, std::string &input, ProcessRequest *request)
{
    if(!state->opened)
    {
        int webcamnumber=0;

        // If they supplied "/dev/video[number]" parse the "number" here
        if(startsWith(input, WEBCAM_PREFIX)&&input.length() > WEBCAM_PREFIX.length())
        {
            webcamnumber=atoi(input.substr(WEBCAM_PREFIX.length()).c_str());
        }

        state->cap.open(webcamnumber);

        if(!state->cap.isOpened())
        {
            std::cerr<<"Error opening webcam"<<std::endl;
            return false;
        }

        state->opened=true;
        state->framenum=0;
    }

    if(!state->cap.read(request->frame))
    {
        std::cerr<<"Error opening webcam"<<std::endl;
        return false;
    }

    if(state->framenum==0)
        motiondetector.ResetMotionDetection(&request->frame);
    state->framenum++;
    return true;
}

bool getUrl(const Settings &settings, UrlState *state, std::string &input, ProcessRequest *request)
{
    if(!state->connected)
    {
        state->videoBuffer.connect(input, 5);
        state->connected=true;
        state->framenum=0;
    }

    std::vector<cv::Rect> regionsOfInterest;
    int response=state->videoBuffer.getLatestFrame(&request->frame, regionsOfInterest);

    if(response==-1)
        return false;

    if(state->framenum==0)
        motiondetector.ResetMotionDetection(&request->frame);
    state->framenum++;
    return true;
}

bool getVideo(const Settings &settings, VideoState *state, std::string &input, ProcessRequest *request)
{
    if(!state->opened)
    {
        if(!fileExists(input.c_str()))
        {
            std::cerr<<"Video file not found: "<<input<<std::endl;
            return false;
        }

        state->framenum=0;

        state->cap.open(input);
        state->cap.set(cv::CAP_PROP_POS_MSEC, settings.seektoms);

        state->opened=true;
    }

    state->cap.read(request->frame);

    if(state->framenum==0)
        motiondetector.ResetMotionDetection(&request->frame);
    state->framenum++;

    return true;
}

bool getImage(const Settings &settings, CustomState *state, std::string &input, ProcessRequest *request)
{
    if(!fileExists(input.c_str()))
    {
        std::cerr<<"Image file not found: "<<input<<std::endl;
        return false;
    }

    request->frame=cv::imread(input);

    if(request->frame.empty())
    {
        std::cerr<<"Image invalid: "<<input<<std::endl;
        return false;
    }
    return true;
}

bool getDirectory(const Settings &settings, DirectoryState *state, std::string &input, ProcessRequest *request)
{
    while(true)
    {
        std::string currentDirectory=state->directoryStack.back();
        std::vector<std::string> files=getFilesInDir(currentDirectory.c_str());
        std::vector<std::string>::iterator fileIter=files.begin();

        if(!state->lastFile.empty())
        {
            fileIter=std::find(files.begin(), files.end(), state->lastFile);

            if(fileIter!=files.end())
                fileIter++;

            if(fileIter==files.end())
            {
                if(state->directoryStack.empty())
                    return false;

                state->lastFile=filenameWithoutExtension(state->directoryStack.back());
                state->directoryStack.pop_back();
                continue;
            }
        }

        std::string file=*fileIter;
        std::string fullpath=currentDirectory+"/"+file;

        state->lastFile=file;

        if(is_supported_image(fullpath))
        {
			if(settings.skipProcessed)
			{
				std::string jsonFile=currentDirectory+"/"+ filenameWithoutExtension(file) +".json";

	            if(fileExists(jsonFile.c_str()))
		            continue;

			}
            request->frame=cv::imread(fullpath.c_str());

            if(request->frame.data==NULL)
                continue;

            request->fileName=fullpath;

            return true;
        }
        else if(DirectoryExists(fullpath.c_str()))
        {
            state->lastFile.clear();
            state->directoryStack.push_back(fullpath);
            continue;
        }
    }

    return false;
}

bool getRequest(const Settings &settings, State &state, ProcessRequest *request)
{
    for(size_t i=state.fileNameIndex; i<state.fileNames.size(); i++)
    {
        std::string fileName=state.fileNames[i];

        if(state.type==MediaType::init)
        {
            state.type=getMediaType(fileName);

            switch(state.type)
            {
            case MediaType::buffer:
                state.stack.emplace_back(new CustomState());
                break;
            case MediaType::stdin_:
                state.stack.emplace_back(new CustomState());
                break;
            case MediaType::webcam:
                state.stack.emplace_back(new VideoState());
                break;
            case MediaType::url:
                state.stack.emplace_back(new UrlState());
                break;
            case MediaType::video:
                state.stack.emplace_back(new VideoState());
                break;
            case MediaType::image:
                state.stack.emplace_back(new CustomState());
                break;
            case MediaType::directory:
                state.stack.emplace_back(new DirectoryState(fileName));
                break;
            default:
                break;
            }
        }

        CustomState *currentState=state.stack.back().get();

        switch(state.type)
        {
        case MediaType::buffer:
            if(getBuffer(settings, currentState, fileName, request))
                return true;
            break;
        case MediaType::stdin_:
            if(getStdIn(settings, currentState, fileName, request))
                return true;
            break;
        case MediaType::webcam:
            if(getWebCam(settings, (VideoState *)currentState, fileName, request))
                return true;
            break;
        case MediaType::url:
            if(getUrl(settings, (UrlState *)currentState, fileName, request))
                return true;
            break;
        case MediaType::video:
            if(getVideo(settings, (VideoState *)currentState, fileName, request))
                return true;
            break;
        case MediaType::image:
            if(getImage(settings, currentState, fileName, request))
                return true;
            break;
        case MediaType::directory:
            if(getDirectory(settings, (DirectoryState *)currentState, fileName, request))
                return true;
            break;
        default:
            std::cerr<<"Unknown file type"<<std::endl;
            return false;
            break;
        }

        state.type==MediaType::init;
        state.fileNameIndex++;
        if(!state.stack.empty())
            state.stack.pop_back();

        return false;
    }
    return false;
}

void processThread(Settings &settings, ThreadInfo *threadInfo)
{
    Alpr alpr(settings.country, settings.configFile);
    cv::Mat frame;

    alpr.setTopN(settings.topn);

    if(settings.debug_mode)
    {
        alpr.getConfig()->setDebug(true);
    }

    if(settings.detectRegion)
        alpr.setDetectRegion(settings.detectRegion);

    if(settings.templatePattern.empty()==false)
        alpr.setDefaultRegion(settings.templatePattern);

    if(alpr.isLoaded()==false)
    {
        std::cerr<<"Error loading OpenALPR"<<std::endl;
        return;
    }

    ProcessRequest *request=nullptr;
    timespec startTime;
    timespec endTime;
    std::vector<AlprRegionOfInterest> regionsOfInterest;
    cv::Rect rectan;

    while(threadInfo->running)
    {
        {
            std::unique_lock<std::mutex> lock(threadInfo->processMutex);

            if(request!=nullptr)
            {
                threadInfo->completed.push_back(request);
                threadInfo->completeEvent.notify_all();
                request=nullptr;
            }

            if(threadInfo->queued.empty())
                threadInfo->processEvent.wait(lock);

            request=threadInfo->queued.front();
            threadInfo->queued.pop_front();
        }

        if(request==nullptr)
            continue;

        regionsOfInterest.clear();

        getTimeMonotonic(&startTime);
        if(settings.do_motiondetection)
        {
            rectan=motiondetector.MotionDetect(&request->frame);
            if(rectan.width>0)
                regionsOfInterest.push_back(AlprRegionOfInterest(rectan.x, rectan.y, rectan.width, rectan.height));
        }
        else
            regionsOfInterest.push_back(AlprRegionOfInterest(0, 0, request->frame.cols, request->frame.rows));

        if(regionsOfInterest.size()>0)
            request->results=alpr.recognize(request->frame.data, request->frame.elemSize(), request->frame.cols, request->frame.rows, regionsOfInterest);

        getTimeMonotonic(&endTime);

        request->processingTime=diffclock(startTime, endTime);
    }

    //we are closeing, return last request if we have one
    if(request!=nullptr)
    {
        std::unique_lock<std::mutex> lock(threadInfo->processMutex);

        threadInfo->completed.push_back(request);
        request=nullptr;
    }
}

//bool processImageDirectory(Alpr &alpr, cv::Mat &frame, std::string &directory, bool outputJson, bool outputJsonFormated)
//{
//    std::vector<std::string> files=getFilesInDir(directory.c_str());
//    bool result=true;
//    std::string jsonFile;
//
//    for(int i=0; i<files.size(); i++)
//    {
//        //		if (processedImages > 10)
//        //			break;
//
//        std::string fullpath=directory+"/"+files[i];
//
//        if(is_supported_image(fullpath))
//        {
//            jsonFile=directory+"/"+filenameWithoutExtension(files[i])+".json";
//
//            if(fileExists(jsonFile.c_str()))
//                continue;
//
//            std::cout<<fullpath<<std::endl;
//            frame=cv::imread(fullpath.c_str());
//
//            if(frame.data==NULL)
//                continue;
//
//            detectandshow(&alpr, frame, "", outputJson, outputJsonFormated, jsonFile);
//        }
//        else if(DirectoryExists(fullpath.c_str()))
//        {
//            processImageDirectory(alpr, frame, fullpath, outputJson, outputJsonFormated);
//        }
//    }
//    return result;
//}

bool is_supported_image(std::string image_file)
{
    return (hasEndingInsensitive(image_file, ".png")||hasEndingInsensitive(image_file, ".jpg")||
        hasEndingInsensitive(image_file, ".tif")||hasEndingInsensitive(image_file, ".bmp")||
        hasEndingInsensitive(image_file, ".jpeg")||hasEndingInsensitive(image_file, ".gif"));
}


//bool detectandshow(Alpr* alpr, cv::Mat frame, std::string region, bool writeJson, bool formated, std::string jsonFile)
//{
//
//    timespec startTime;
//    getTimeMonotonic(&startTime);
//
//    std::vector<AlprRegionOfInterest> regionsOfInterest;
//    if(do_motiondetection)
//    {
//        cv::Rect rectan=motiondetector.MotionDetect(&frame);
//        if(rectan.width>0) regionsOfInterest.push_back(AlprRegionOfInterest(rectan.x, rectan.y, rectan.width, rectan.height));
//    }
//    else regionsOfInterest.push_back(AlprRegionOfInterest(0, 0, frame.cols, frame.rows));
//    AlprResults results;
//    if(regionsOfInterest.size()>0) results=alpr->recognize(frame.data, frame.elemSize(), frame.cols, frame.rows, regionsOfInterest);
//
//    timespec endTime;
//    getTimeMonotonic(&endTime);
//    double totalProcessingTime=diffclock(startTime, endTime);
//    if(measureProcessingTime)
//        std::cout<<"Total Time to process image: "<<totalProcessingTime<<"ms."<<std::endl;
//
//    processedImages++;
//
//    if(writeJson)
//    {
//        std::string json=toJson(results, formated);
//        std::cout<<json<<std::endl;
//
//        if(!jsonFile.empty())
//        {
//            FILE *file=fopen(jsonFile.c_str(), "w");
//
//            if(file)
//            {
//                fwrite(json.c_str(), sizeof(char), json.size(), file);
//                fclose(file);
//            }
//        }
//    }
//    else
//    {
//        for(int i=0; i < results.plates.size(); i++)
//        {
//            std::cout<<"plate"<<i<<": "<<results.plates[i].topNPlates.size()<<" results";
//            if(measureProcessingTime)
//                std::cout<<" -- Processing Time = "<<results.plates[i].processing_time_ms<<"ms.";
//            std::cout<<std::endl;
//
//            if(results.plates[i].regionConfidence > 0)
//                std::cout<<"State ID: "<<results.plates[i].region<<" ("<<results.plates[i].regionConfidence<<"% confidence)"<<std::endl;
//
//            for(int k=0; k < results.plates[i].topNPlates.size(); k++)
//            {
//                // Replace the multiline newline character with a dash
//                std::string no_newline=results.plates[i].topNPlates[k].characters;
//                std::replace(no_newline.begin(), no_newline.end(), '\n', '-');
//
//                std::cout<<"    - "<<no_newline<<"\t confidence: "<<results.plates[i].topNPlates[k].overall_confidence;
//                if(settings.templatePattern.size() > 0||results.plates[i].regionConfidence > 0)
//                    std::cout<<"\t pattern_match: "<<results.plates[i].topNPlates[k].matches_template;
//
//                std::cout<<std::endl;
//            }
//        }
//    }
//
//
//
//    return results.plates.size() > 0;
//}

cJSON* toJsonObj(const AlprPlateResult* result)
{
    cJSON *root, *coords, *candidates, *characters;

    root=cJSON_CreateObject();

    cJSON_AddStringToObject(root, "plate", result->bestPlate.characters.c_str());
    cJSON_AddNumberToObject(root, "confidence", result->bestPlate.overall_confidence);
    cJSON_AddNumberToObject(root, "matches_template", result->bestPlate.matches_template);

    cJSON_AddNumberToObject(root, "plate_index", result->plate_index);

    cJSON_AddStringToObject(root, "region", result->region.c_str());
    cJSON_AddNumberToObject(root, "region_confidence", result->regionConfidence);

    cJSON_AddNumberToObject(root, "processing_time_ms", result->processing_time_ms);
    cJSON_AddNumberToObject(root, "requested_topn", result->requested_topn);

    cJSON_AddItemToObject(root, "coordinates", coords=cJSON_CreateArray());
    for(int i=0; i < 4; i++)
    {
        cJSON *coords_object;
        coords_object=cJSON_CreateObject();
        cJSON_AddNumberToObject(coords_object, "x", result->plate_points[i].x);
        cJSON_AddNumberToObject(coords_object, "y", result->plate_points[i].y);

        cJSON_AddItemToArray(coords, coords_object);
    }


    cJSON_AddItemToObject(root, "candidates", candidates=cJSON_CreateArray());
    for(unsigned int i=0; i < result->topNPlates.size(); i++)
    {
        cJSON *candidate_object;
        const AlprPlate &plate=result->topNPlates[i];

        candidate_object=cJSON_CreateObject();
        cJSON_AddStringToObject(candidate_object, "plate", plate.characters.c_str());
        cJSON_AddNumberToObject(candidate_object, "confidence", plate.overall_confidence);
        cJSON_AddNumberToObject(candidate_object, "matches_template", plate.matches_template);

        cJSON_AddItemToObject(candidate_object, "characters", characters=cJSON_CreateArray());
        for(unsigned int j=0; j < plate.character_details.size(); j++)
        {
            cJSON *char_object, *coords;
            const AlprChar &character_details=plate.character_details[j];

            char_object=cJSON_CreateObject();

            cJSON_AddStringToObject(char_object, "character", character_details.character.c_str());
            cJSON_AddNumberToObject(char_object, "confidence", character_details.confidence);

            cJSON_AddItemToObject(char_object, "coordinates", coords=cJSON_CreateArray());
            for(int k=0; k < 4; k++)
            {
                cJSON *coords_object;
                const AlprCoordinate &alpr_coords=character_details.corners[k];

                coords_object=cJSON_CreateObject();
                cJSON_AddNumberToObject(coords_object, "x", alpr_coords.x);
                cJSON_AddNumberToObject(coords_object, "y", alpr_coords.y);

                cJSON_AddItemToArray(coords, coords_object);
            }
            cJSON_AddItemToArray(characters, char_object);
        }

        cJSON_AddItemToArray(candidates, candidate_object);
    }

    return root;
}

std::string toJson(const AlprResults results, bool formated)
{
    cJSON *root, *jsonResults;
    root=cJSON_CreateObject();


    cJSON_AddNumberToObject(root, "version", 2);
    cJSON_AddStringToObject(root, "data_type", "alpr_results");

    cJSON_AddNumberToObject(root, "epoch_time", results.epoch_time);
    cJSON_AddNumberToObject(root, "img_width", results.img_width);
    cJSON_AddNumberToObject(root, "img_height", results.img_height);
    cJSON_AddNumberToObject(root, "processing_time_ms", results.total_processing_time_ms);

    // Add the regions of interest to the JSON
    cJSON *rois;
    cJSON_AddItemToObject(root, "regions_of_interest", rois=cJSON_CreateArray());
    for(unsigned int i=0; i < results.regionsOfInterest.size(); i++)
    {
        cJSON *roi_object;
        roi_object=cJSON_CreateObject();
        cJSON_AddNumberToObject(roi_object, "x", results.regionsOfInterest[i].x);
        cJSON_AddNumberToObject(roi_object, "y", results.regionsOfInterest[i].y);
        cJSON_AddNumberToObject(roi_object, "width", results.regionsOfInterest[i].width);
        cJSON_AddNumberToObject(roi_object, "height", results.regionsOfInterest[i].height);

        cJSON_AddItemToArray(rois, roi_object);
    }


    cJSON_AddItemToObject(root, "results", jsonResults=cJSON_CreateArray());
    for(unsigned int i=0; i < results.plates.size(); i++)
    {
        cJSON *resultObj=toJsonObj(&results.plates[i]);
        cJSON_AddItemToArray(jsonResults, resultObj);
    }

    // Print the JSON object to a string and return
    char *out;

    if(formated)
        out=cJSON_Print(root);
    else
        out=cJSON_PrintUnformatted(root);

    cJSON_Delete(root);

    std::string response(out);

    free(out);
    return response;
}
