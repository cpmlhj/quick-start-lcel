import { DocumentInterface } from '@langchain/core/documents'
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter'
import { PlaywrightWebBaseLoader } from '@langchain/community/document_loaders/web/playwright'
import { compile } from 'html-to-text'
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai'
import { Chroma } from '@langchain/community/vectorstores/chroma'
import dot from 'dotenv'
import { HttpsProxyAgent } from 'https-proxy-agent'
import { pull } from 'langchain/hub'
import {
	RunnablePassthrough,
	RunnableSequence
} from '@langchain/core/runnables'
import { StringOutputParser } from '@langchain/core/output_parsers'
import { ChatPromptTemplate } from '@langchain/core/prompts'

import fs from 'fs'

dot.config()

const hta = new HttpsProxyAgent(process.env.HTTP_PROXY || '')

function formatDocs(docs: DocumentInterface[]) {
	let stringDoc = ''
	docs.forEach((doc) => {
		stringDoc += doc.pageContent + '\n\n'
	})
	return stringDoc
}

async function main() {
	/**
	 * RAG 应用
	 * 1.获取、加载文档
	 * 2.分割文档
	 * 3.存储嵌入
	 * 4.向量检索
	 * 5.生成答案
	 */
	try {
		//  1.获取、加载文档
		const loader = new PlaywrightWebBaseLoader(
			'https://hackaday.io/page/399365-writing-a-z80-disassembler-using-ai',
			{
				launchOptions: {
					headless: true,
					timeout: 100000
				},
				gotoOptions: {
					waitUntil: 'networkidle',
					timeout: 100000
				},
				evaluate: async (page) => {
					await page.evaluate(() => {
						window.scrollTo(0, document.body.scrollHeight)
					})

					await page.waitForTimeout(2000)

					const content = await page.evaluate(() => {
						const content = document.body.innerText

						return content
					})
					return content || ''
				}
			}
		)
		const docs = await loader.load()

		const spiltter = new RecursiveCharacterTextSplitter({
			chunkSize: 500,
			chunkOverlap: 200
		})

		// 转换、分隔文档
		const spilt_content = await spiltter.splitDocuments(docs)
		// 存储文档
		const embddingModel = new OpenAIEmbeddings({
			configuration: {
				httpAgent: hta
			}
		})
		const store = new Chroma(embddingModel, {
			collectionName: 'langchain'
		})
		await store.addDocuments(spilt_content)

		// 向量检索
		const retriever = store.asRetriever({ k: 2 })

		// 生成答案
		// const remote_prompt = await pull('rlm/rag-prompt')
		const file_prompt = fs.readFileSync('./prompt.md', 'utf-8')
		const template = ChatPromptTemplate.fromTemplate(file_prompt)
		const chatmodel = new ChatOpenAI({
			model: 'gpt-4',
			configuration: {
				httpAgent: hta
			}
		})

		const chain = RunnableSequence.from([
			RunnablePassthrough.assign({
				question: (input) => input.input,
				context: async (input) => {
					const context = await retriever.invoke(
						input.input as string
					)
					return formatDocs(context)
				}
			}),
			template,
			chatmodel,
			new StringOutputParser()
		])

		const stream = await chain.stream({
			input: '金三顺来了'
		})

		for await (const chunk of stream) {
			process.stdout.write(chunk)
		}
	} catch (error) {
		console.log(error)
	}
}

main()
