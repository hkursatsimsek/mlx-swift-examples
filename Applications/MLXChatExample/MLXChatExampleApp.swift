//
//  MLXChatExampleApp.swift
//  MLXChatExample
//
//  Created by İbrahim Çetin on 20.04.2025.
//

import Metal
import SwiftUI

@main
struct MLXChatExampleApp: App {
    // MLX requires Apple GPU Family 7+ (A14 Bionic / iPhone 12 minimum)
    private static let isGPUFamilySupported: Bool = {
        guard let device = MTLCreateSystemDefaultDevice() else { return false }
        return device.supportsFamily(.apple7)
    }()

    var body: some Scene {
        WindowGroup {
            if Self.isGPUFamilySupported {
                ChatView(viewModel: ChatViewModel(mlxService: MLXService()))
            } else {
                UnsupportedDeviceView()
            }
        }
    }
}
