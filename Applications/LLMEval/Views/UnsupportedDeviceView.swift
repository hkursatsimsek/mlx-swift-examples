// Copyright © 2025 Apple Inc.

import SwiftUI

struct UnsupportedDeviceView: View {
    var body: some View {
        VStack(spacing: 24) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 64))
                .foregroundStyle(.orange)

            VStack(spacing: 8) {
                Text("Desteklenmeyen Cihaz")
                    .font(.title2.bold())

                Text("Bu uygulama Apple GPU Family 7 veya daha yeni bir GPU gerektiriyor.")
                    .font(.body)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }

            VStack(alignment: .leading, spacing: 8) {
                Label("Desteklenen minimum cihazlar:", systemImage: "iphone")
                    .font(.subheadline.bold())

                Text("• iPhone 12 (A14 Bionic)\n• iPad Pro 2021 (M1)\n• Mac Apple Silicon")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            .padding()
            .background(.fill.tertiary, in: RoundedRectangle(cornerRadius: 12))
        }
        .padding(32)
    }
}
