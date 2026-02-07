
#include <jni.h>
#include <string>
#include <vector>
#include <memory>
// #include <torch/script.h> // Will be available in Android build environment

#include "game.h"
#include "mcts.h"

// Global game state (simplification for single game instance)
std::unique_ptr<GameState> global_state;
// torch::jit::script::Module global_model; // Uncomment when LibTorch is linked

extern "C" JNIEXPORT jstring JNICALL
Java_com_alphaludo_app_NativeLib_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from AlphaLudo C++";
    return env->NewStringUTF(hello.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_alphaludo_app_NativeLib_initGame(
        JNIEnv* env,
        jobject /* this */,
        jstring modelPath) {
    
    // Initialize new game
    global_state = std::make_unique<GameState>();
    
    // Load model
    const char *path = env->GetStringUTFChars(modelPath, 0);
    // try {
    //     global_model = torch::jit::load(path);
    // } catch (const c10::Error& e) {
    //     // Handle error
    // }
    env->ReleaseStringUTFChars(modelPath, path);
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_alphaludo_app_NativeLib_applyMove(
        JNIEnv* env,
        jobject /* this */,
        jint action) {
    
    if (!global_state) return false;
    
    // Apply move using game.cpp logic
    *global_state = apply_move(*global_state, action);
    
    return true;
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_alphaludo_app_NativeLib_getLegalMoves(
        JNIEnv* env,
        jobject /* this */) {
            
    if (!global_state) return env->NewIntArray(0);

    std::vector<int> moves = get_legal_moves(*global_state);
    
    jintArray result = env->NewIntArray(moves.size());
    env->SetIntArrayRegion(result, 0, moves.size(), moves.data());
    return result;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_alphaludo_app_NativeLib_getCurrentPlayer(
        JNIEnv* env,
        jobject /* this */) {
    if (!global_state) return -1;
    return global_state->current_player;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_alphaludo_app_NativeLib_getDiceRoll(
        JNIEnv* env,
        jobject /* this */) {
    if (!global_state) return 0;
    return global_state->current_dice_roll;
}

// Function to get board state for UI rendering
// Returns flat array: [p0_pos..., p1_pos..., scores..., winner]
extern "C" JNIEXPORT jintArray JNICALL
Java_com_alphaludo_app_NativeLib_getBoardState(
        JNIEnv* env,
        jobject /* this */) {
            
    if (!global_state) return env->NewIntArray(0);

    std::vector<int> data;
    
    // Positions (4 players * 4 tokens = 16)
    for(int p=0; p<4; ++p) {
        for(int t=0; t<4; ++t) {
            data.push_back(global_state->player_positions[p][t]);
        }
    }
    
    // Scores (4)
    for(int p=0; p<4; ++p) {
        data.push_back(global_state->scores[p]);
    }
    
    // Winner check
    int winner = -1;
    for(int p=0; p<4; ++p) {
        if (global_state->scores[p] == 4) {
            winner = p;
            break;
        }
    }
    data.push_back(winner); // Last element is winner
    
    jintArray result = env->NewIntArray(data.size());
    env->SetIntArrayRegion(result, 0, data.size(), data.data());
    return result;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_alphaludo_app_NativeLib_getModelMove(
        JNIEnv* env,
        jobject /* this */) {
            
    // 1. Convert state to tensor
    // 2. Run inference via global_model
    // 3. Mask illegal moves
    // 4. Return best action
    
    return -1; // Placeholder
}
