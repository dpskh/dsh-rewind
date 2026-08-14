import type { UserConfig } from 'tsdown'

export default {
  name: '@dpskh/tool-rewind',
  entry: ['lib/types/index.js', 'lib/types/invariant.js'],
  outDir: 'lib',
  format: 'esm',
  platform: 'node',
  target: 'node22',
  dts: false,
  clean: false,
  outputOptions: {
    entryFileNames: '[name].js',
  },
} satisfies UserConfig
